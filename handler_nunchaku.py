"""Persistent 24 GB Runpod worker for the community Nunchaku Qwen 2511 build."""

from __future__ import annotations

import base64
import binascii
import gc
import io
import math
import os
import secrets
import threading
import time
from typing import Any

import runpod
from PIL import Image

MAX_IMAGE_BYTES = 15 * 1024 * 1024
MAX_IMAGE_PIXELS = 24_000_000
MAX_IMAGES = 2
MAX_OUTPUT_PIXELS = 1_048_576
JOB_LOCK = threading.Lock()
MODEL_REPO = "tonera/Qwen-Image-Edit-2511-Lightning-Nunchaku"
TEXT_ENCODER_REPO = "tonera/Qwen2.5vl-Nunchaku"
MODEL_NAME = "Qwen-Image-Edit-2511-Lightning-Nunchaku"


def decode_image(value: Any) -> Image.Image:
    if not isinstance(value, str) or not value.startswith("data:image/"):
        raise ValueError("each image must be a base64 image data URL")
    try:
        _, encoded = value.split(",", 1)
        raw = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValueError("image must contain valid base64 data") from exc
    if not raw or len(raw) > MAX_IMAGE_BYTES:
        raise ValueError("each image must be between 1 byte and 15 MiB")
    try:
        image = Image.open(io.BytesIO(raw))
        image.load()
        if image.width * image.height > MAX_IMAGE_PIXELS:
            raise ValueError("each image must not exceed 24 megapixels")
        return image.convert("RGB")
    except (OSError, ValueError) as exc:
        raise ValueError("each image must be a valid JPEG, PNG, or WebP") from exc


def output_size(image: Image.Image) -> tuple[int, int]:
    scale = min(1.0, math.sqrt(MAX_OUTPUT_PIXELS / (image.width * image.height)))
    return (
        max(16, int(image.width * scale) // 16 * 16),
        max(16, int(image.height * scale) // 16 * 16),
    )


def encode_image(image: Image.Image) -> str:
    data = io.BytesIO()
    image.save(data, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(data.getvalue()).decode("ascii")


class NunchakuWorker:
    def __init__(self) -> None:
        try:
            import torch
            from diffusers import QwenImageEditPlusPipeline
            from huggingface_hub import hf_hub_download
            from nunchaku import NunchakuQwenEncoderModel, NunchakuQwenImageTransformer2DModel
            from nunchaku.torch_transfer_utils import pretouch_pipeline_cpu_tensors
            from nunchaku.utils import get_precision
        except Exception as exc:
            raise RuntimeError(f"Nunchaku imports failed: {type(exc).__name__}: {exc}") from exc
        if not torch.cuda.is_available():
            raise RuntimeError("Nunchaku requires a CUDA GPU")

        self.torch = torch
        precision = os.getenv("NUNCHAKU_PRECISION") or get_precision()
        if precision not in {"int4", "fp4"}:
            raise RuntimeError("NUNCHAKU_PRECISION must be int4 or fp4")
        # This community fork's from_pretrained expects filesystem paths (it
        # does not resolve Hugging Face repo/file strings itself). hf_hub_download
        # resolves the already-populated shared-volume cache without redownloading.
        transformer_path = hf_hub_download(
            MODEL_REPO,
            filename=f"svdq-{precision}_r32-{MODEL_NAME}.safetensors",
        )
        text_encoder_path = hf_hub_download(
            TEXT_ENCODER_REPO,
            filename="svdq-int4-Qwen2.5vl-Nunchaku.safetensors",
        )
        print(
            f"[qwen-nunchaku] loading {precision} Lightning model on "
            f"{torch.cuda.get_device_name(0)!r}",
            flush=True,
        )
        text_encoder = NunchakuQwenEncoderModel.from_pretrained(text_encoder_path)
        transformer = NunchakuQwenImageTransformer2DModel.from_pretrained(transformer_path)
        self.pipeline = QwenImageEditPlusPipeline.from_pretrained(
            MODEL_REPO,
            text_encoder=text_encoder,
            transformer=transformer,
            torch_dtype=torch.bfloat16,
        )
        memory_mode = os.getenv("NUNCHAKU_MEMORY_MODE", "model_cpu_offload")
        if memory_mode == "resident":
            # Useful on larger GPUs. Pre-touching avoids page faults only when
            # every module will immediately become GPU-resident.
            pretouch_pipeline_cpu_tensors(
                self.pipeline, ("text_encoder", "text_encoder_2", "vae", "unet", "transformer")
            )
            self.pipeline.to("cuda")
        elif memory_mode == "model_cpu_offload":
            # Nunchaku documents this configuration for GPUs with >18 GB. It
            # retains the quantized model path but moves VAE/encoder components
            # between phases so 22 GB usable VRAM has decode headroom. Do not
            # pre-touch all tensors here: it delays cold start by minutes and
            # does not help this sequential transfer mode.
            self.pipeline.enable_model_cpu_offload()
        else:
            raise RuntimeError("NUNCHAKU_MEMORY_MODE must be resident or model_cpu_offload")
        self.precision = precision
        self.memory_mode = memory_mode
        self._cleanup()
        print(f"[qwen-nunchaku] model ready; memory_mode={memory_mode}", flush=True)

    def _cleanup(self) -> dict[str, float]:
        self.torch.cuda.synchronize()
        gc.collect()
        self.torch.cuda.empty_cache()
        free, total = self.torch.cuda.mem_get_info("cuda")
        return {
            "free_gb": round(free / 1024**3, 2),
            "total_gb": round(total / 1024**3, 2),
            "allocated_gb": round(self.torch.cuda.memory_allocated() / 1024**3, 2),
            "reserved_gb": round(self.torch.cuda.memory_reserved() / 1024**3, 2),
        }

    def edit(self, payload: dict[str, Any]) -> dict[str, Any]:
        values = payload.get("images")
        if values is None:
            values = [payload.get("image")]
        if not isinstance(values, list) or not 1 <= len(values) <= MAX_IMAGES:
            raise ValueError("provide image, or images containing one or two images")
        images = [decode_image(value) for value in values]
        prompt = payload.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 2_000:
            raise ValueError("prompt must be a non-empty string no longer than 2000 characters")
        seed = payload.get("seed", secrets.randbelow(2**31))
        if not isinstance(seed, int) or not 0 <= seed < 2**32:
            raise ValueError("seed must be an integer between 0 and 4294967295")
        steps = payload.get("steps", 4)
        if not isinstance(steps, int) or not 4 <= steps <= 8:
            raise ValueError("steps must be an integer from 4 to 8")

        width, height = output_size(images[0])
        started = time.monotonic()
        output = None
        try:
            with self.torch.inference_mode():
                output = self.pipeline(
                    prompt=prompt.strip(),
                    negative_prompt=" ",
                    image=images[0] if len(images) == 1 else images,
                    width=width,
                    height=height,
                    num_inference_steps=steps,
                    true_cfg_scale=1.0,
                    guidance_scale=1.0,
                    generator=self.torch.Generator("cuda").manual_seed(seed),
                ).images[0]
            return {
                "provider": "qwen-image-edit-2511-lightning-nunchaku",
                "precision": self.precision,
                "memory_mode": self.memory_mode,
                "seed": seed,
                "width": output.width,
                "height": output.height,
                "execution_seconds": round(time.monotonic() - started, 2),
                "image": encode_image(output),
            }
        finally:
            del output
            stats = self._cleanup()
            print(f"[qwen-nunchaku] request cleanup: {stats}", flush=True)


WORKER: NunchakuWorker | None = None


def get_worker() -> NunchakuWorker:
    global WORKER
    if WORKER is None:
        WORKER = NunchakuWorker()
    return WORKER


def handler(job: dict[str, Any]) -> dict[str, Any]:
    try:
        payload = job.get("input")
        if not isinstance(payload, dict):
            raise ValueError("input must be a JSON object")
        with JOB_LOCK:
            return get_worker().edit(payload)
    except ValueError as exc:
        return {"error": str(exc)}
    except Exception as exc:
        detail = f"{type(exc).__name__}: {exc}"
        print(f"[qwen-nunchaku] inference error: {detail}", flush=True)
        return {"error": "Nunchaku Qwen Image Edit inference failed", "detail": detail}


print("[qwen-nunchaku] Runpod handler booted; model will load on first job", flush=True)
runpod.serverless.start({"handler": handler})
