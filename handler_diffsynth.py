"""Runpod worker for Qwen Image Edit 2511 without ComfyUI.

DiffSynth's VRAM manager loads only the components needed by each stage of an
edit.  Unlike a persistent sd-server, it ends every request with all modules
offloaded, so a sequence of jobs cannot retain the previous image's working
buffers on a 24 GB GPU.
"""

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
DEFAULT_MAX_OUTPUT_PIXELS = 1_048_576
JOB_LOCK = threading.Lock()


def _positive_int_env(name: str, default: int) -> int:
    value = os.getenv(name, str(default))
    try:
        parsed = int(value)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer") from exc
    if parsed <= 0:
        raise RuntimeError(f"{name} must be positive")
    return parsed


def _decode_image(value: Any) -> Image.Image:
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


def _output_size(image: Image.Image) -> tuple[int, int]:
    """Keep source aspect ratio and align the Qwen canvas to multiples of 16."""
    limit = _positive_int_env("QWEN_MAX_OUTPUT_PIXELS", DEFAULT_MAX_OUTPUT_PIXELS)
    scale = min(1.0, math.sqrt(limit / (image.width * image.height)))
    width = max(16, int(image.width * scale) // 16 * 16)
    height = max(16, int(image.height * scale) // 16 * 16)
    return width, height


def _encode_image(image: Image.Image) -> str:
    data = io.BytesIO()
    image.save(data, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(data.getvalue()).decode("ascii")


class QwenImageEditWorker:
    def __init__(self) -> None:
        try:
            import torch
            from diffsynth.core import ModelConfig
            from diffsynth.diffusion import FlowMatchScheduler
            from diffsynth.pipelines.qwen_image import QwenImagePipeline
        except Exception as exc:
            raise RuntimeError(f"DiffSynth imports failed: {type(exc).__name__}: {exc}") from exc
        if not torch.cuda.is_available():
            raise RuntimeError("Qwen Image Edit requires a CUDA GPU")

        self.torch = torch
        self.ModelConfig = ModelConfig
        self.max_vram_gb = self._vram_limit()
        dtype = self._offload_dtype()
        vram_config = {
            "offload_dtype": "disk",
            "offload_device": "disk",
            "onload_dtype": "disk",
            "onload_device": "disk",
            "preparing_dtype": dtype,
            "preparing_device": "cuda",
            "computation_dtype": torch.bfloat16,
            "computation_device": "cuda",
        }
        print(
            "[qwen-diffsynth] loading Qwen Image Edit 2511; "
            f"gpu={torch.cuda.get_device_name(0)!r}, vram_limit={self.max_vram_gb:.1f}GB, "
            f"stage_dtype={dtype}",
            flush=True,
        )
        self.pipeline = QwenImagePipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device="cuda",
            model_configs=[
                ModelConfig(
                    model_id="Qwen/Qwen-Image-Edit-2511",
                    origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors",
                    **vram_config,
                ),
                ModelConfig(
                    model_id="Qwen/Qwen-Image",
                    origin_file_pattern="text_encoder/model*.safetensors",
                    **vram_config,
                ),
                ModelConfig(
                    model_id="Qwen/Qwen-Image",
                    origin_file_pattern="vae/diffusion_pytorch_model.safetensors",
                    **vram_config,
                ),
            ],
            processor_config=ModelConfig(
                model_id="Qwen/Qwen-Image-Edit", origin_file_pattern="processor/"
            ),
            vram_limit=self.max_vram_gb,
        )
        lora = ModelConfig(
            model_id="lightx2v/Qwen-Image-Edit-2511-Lightning",
            origin_file_pattern="Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors",
        )
        self.pipeline.load_lora(self.pipeline.dit, lora, alpha=1.0)
        self.pipeline.scheduler = FlowMatchScheduler("Qwen-Image-Lightning")
        self.release_gpu_memory()
        print("[qwen-diffsynth] model ready; all inactive modules offloaded", flush=True)

    def _offload_dtype(self) -> Any:
        mode = os.getenv("QWEN_STAGE_DTYPE", "bf16").lower()
        if mode == "bf16":
            return self.torch.bfloat16
        if mode == "fp8":
            return self.torch.float8_e4m3fn
        raise RuntimeError("QWEN_STAGE_DTYPE must be bf16 or fp8")

    def _vram_limit(self) -> float:
        total_gb = self.torch.cuda.mem_get_info("cuda")[1] / 1024**3
        configured = os.getenv("QWEN_VRAM_LIMIT_GB")
        if configured is not None:
            value = float(configured)
            if value <= 0 or value >= total_gb:
                raise RuntimeError(
                    f"QWEN_VRAM_LIMIT_GB must be > 0 and smaller than physical VRAM ({total_gb:.1f})"
                )
            return value
        # Leave headroom for input latents, VAE activation and output encoding.
        return max(1.0, total_gb - 2.0)

    def release_gpu_memory(self) -> dict[str, float]:
        """Return every DiffSynth module to offload state, then clear request buffers."""
        self.pipeline.load_models_to_device([])
        self.torch.cuda.synchronize()
        gc.collect()
        self.torch.cuda.empty_cache()
        if hasattr(self.torch.cuda, "ipc_collect"):
            self.torch.cuda.ipc_collect()
        free, total = self.torch.cuda.mem_get_info("cuda")
        return {
            "free_gb": round(free / 1024**3, 2),
            "total_gb": round(total / 1024**3, 2),
            "allocated_gb": round(self.torch.cuda.memory_allocated() / 1024**3, 2),
            "reserved_gb": round(self.torch.cuda.memory_reserved() / 1024**3, 2),
        }

    def edit(self, payload: dict[str, Any]) -> dict[str, Any]:
        image_values = payload.get("images")
        if image_values is None:
            image_values = [payload.get("image")]
        if not isinstance(image_values, list) or not 1 <= len(image_values) <= MAX_IMAGES:
            raise ValueError("provide image, or images containing one or two images")
        images = [_decode_image(value) for value in image_values]
        prompt = payload.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 2_000:
            raise ValueError("prompt must be a non-empty string no longer than 2000 characters")
        seed = payload.get("seed", secrets.randbelow(2**31))
        if not isinstance(seed, int) or not 0 <= seed < 2**32:
            raise ValueError("seed must be an integer between 0 and 4294967295")
        steps = payload.get("steps", 4)
        if steps != 4:
            raise ValueError("this Lightning endpoint requires steps=4")

        width, height = _output_size(images[0])
        started = time.monotonic()
        try:
            with self.torch.inference_mode():
                output = self.pipeline(
                    prompt.strip(),
                    edit_image=images,
                    seed=seed,
                    num_inference_steps=4,
                    height=height,
                    width=width,
                    edit_image_auto_resize=True,
                    zero_cond_t=True,
                    cfg_scale=1.0,
                )
            return {
                "provider": "qwen-image-edit-2511-lightning-diffsynth",
                "seed": seed,
                "width": output.width,
                "height": output.height,
                "execution_seconds": round(time.monotonic() - started, 2),
                "image": _encode_image(output),
            }
        finally:
            stats = self.release_gpu_memory()
            print(f"[qwen-diffsynth] request cleanup: {stats}", flush=True)


WORKER: QwenImageEditWorker | None = None


def _worker() -> QwenImageEditWorker:
    global WORKER
    if WORKER is None:
        WORKER = QwenImageEditWorker()
    return WORKER


def handler(job: dict[str, Any]) -> dict[str, Any]:
    try:
        payload = job.get("input")
        if not isinstance(payload, dict):
            raise ValueError("input must be a JSON object")
        # A Runpod worker is normally single-job, but this also prevents future
        # local/API concurrency from overlapping two VRAM-heavy pipelines.
        with JOB_LOCK:
            return _worker().edit(payload)
    except ValueError as exc:
        return {"error": str(exc)}
    except Exception as exc:
        detail = f"{type(exc).__name__}: {exc}"
        print(f"[qwen-diffsynth] inference error: {detail}", flush=True)
        return {"error": "Qwen Image Edit inference failed", "detail": detail}


print("[qwen-diffsynth] Runpod handler booted; pipeline will load on first job", flush=True)
runpod.serverless.start({"handler": handler})
