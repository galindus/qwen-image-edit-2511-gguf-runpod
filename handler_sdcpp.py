"""Runpod handler backed by one persistent stable-diffusion.cpp server.

Input:
  {"image": "data:image/...;base64,...", "prompt": "...", "seed": 42}

`image` may also be raw base64.  Width and height default to the input image,
bounded by SDC_MAX_SIDE while preserving its aspect ratio.
"""

import base64
import io
import json
import os
import shlex
import subprocess
import threading
import time
from typing import Any

import requests
import runpod
from PIL import Image


MODEL_ROOT = os.environ["MODEL_ROOT"]
PORT = int(os.environ["SDC_PORT"])
BASE_URL = f"http://127.0.0.1:{PORT}"
SERVER_LOCK = threading.Lock()
INFERENCE_LOCK = threading.Lock()
SERVER: subprocess.Popen | None = None


def _model_path(directory: str, filename: str) -> str:
    path = os.path.join(MODEL_ROOT, directory, filename)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"missing model file: {path}")
    return path


def _start_server() -> None:
    """Start and load sd-server lazily, so the Runpod worker boots quickly."""
    global SERVER
    with SERVER_LOCK:
        if SERVER and SERVER.poll() is None:
            return

        command = [
            "/sd-server",
            "--diffusion-model", _model_path("diffusion_models", os.environ["SDC_TRANSFORMER"]),
            "--vae", _model_path("vae", os.environ["SDC_VAE"]),
            "--llm", _model_path("text_encoders", os.environ["SDC_TEXT_ENCODER"]),
            "--lora-model-dir", os.path.join(MODEL_ROOT, "loras"),
            "--diffusion-fa",
            "--model-args", "qwen_image_zero_cond_t=true",
            "--listen-ip", "127.0.0.1",
            "--listen-port", str(PORT),
            "--verbose",
        ]
        # The GGUF Qwen2.5-VL encoder needs its mmproj.  ComfyUI's FP8
        # qwen_2.5_vl_7b encoder already contains the vision component, so
        # set SDC_MMPROJ=none to reproduce that configuration.
        mmproj = os.environ.get("SDC_MMPROJ", "")
        if mmproj.lower() not in {"", "0", "false", "none"}:
            command.extend(["--llm_vision", _model_path("text_encoders", mmproj)])

        # Keep the 14 GB Q4 DiT resident on GPU, but run the Qwen2.5-VL
        # conditioner on CPU. It only runs once per request; this frees ~6 GB
        # through denoising and VAE decode, avoiding the accumulation observed
        # with a fully-resident 24 GB worker. `stream` is retained only as an
        # experimental option: Qwen Edit 2511 currently crashes with ggml graph
        # cuts / --max-vram.
        memory_mode = os.environ.get("SDC_MEMORY_MODE", "clip_cpu").lower()
        if memory_mode == "clip_cpu":
            command.extend(
                [
                    "--backend", "diffusion=cuda0,te=cpu,clip_vision=cpu,vae=cuda0",
                    "--params-backend", "diffusion=cuda0,te=cpu,clip_vision=cpu,vae=cuda0",
                ]
            )
        elif memory_mode == "clip_vae_cpu":
            command.extend(
                [
                    "--backend", "diffusion=cuda0,te=cpu,clip_vision=cpu,vae=cpu",
                    "--params-backend", "diffusion=cuda0,te=cpu,clip_vision=cpu,vae=cpu",
                ]
            )
        elif memory_mode == "stream":
            command.extend(["--offload-to-cpu", "--max-vram", "-1", "--stream-layers"])
        elif memory_mode != "resident":
            raise ValueError(
                "SDC_MEMORY_MODE must be resident, clip_cpu, clip_vae_cpu, or stream"
            )

        extra_args = os.environ.get("SDC_EXTRA_ARGS", "")
        if extra_args:
            command.extend(shlex.split(extra_args))
        print("[sdcpp] starting persistent sd-server", flush=True)
        SERVER = subprocess.Popen(command)

        deadline = time.monotonic() + 15 * 60
        while time.monotonic() < deadline:
            if SERVER.poll() is not None:
                raise RuntimeError(f"sd-server exited during startup ({SERVER.returncode})")
            try:
                response = requests.get(f"{BASE_URL}/sdcpp/v1/capabilities", timeout=2)
                response.raise_for_status()
                print("[sdcpp] model server ready", flush=True)
                return
            except requests.RequestException:
                time.sleep(1)
        raise TimeoutError("sd-server did not become ready within 15 minutes")


def _normalise_image(image: str) -> tuple[str, int, int]:
    raw_b64 = image.split(",", 1)[1] if image.startswith("data:") else image
    binary = base64.b64decode(raw_b64, validate=True)
    with Image.open(io.BytesIO(binary)) as source:
        width, height = source.size

    max_side = int(os.environ["SDC_MAX_SIDE"])
    scale = min(1.0, max_side / max(width, height))
    # Qwen's latent grid needs dimensions divisible by 16.
    target_width = max(16, round(width * scale / 16) * 16)
    target_height = max(16, round(height * scale / 16) * 16)
    return raw_b64, target_width, target_height


def _request_payload(job_input: dict[str, Any]) -> dict[str, Any]:
    image = job_input.get("image") or job_input.get("input_image")
    prompt = job_input.get("prompt")
    if not isinstance(image, str) or not image:
        raise ValueError("input.image (base64 or data URL) is required")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("input.prompt is required")

    image_b64, width, height = _normalise_image(image)
    width = int(job_input.get("width", width))
    height = int(job_input.get("height", height))
    if width % 16 or height % 16:
        raise ValueError("width and height must be divisible by 16")

    steps = int(job_input.get("steps", os.environ["SDC_STEPS"]))
    cfg = float(job_input.get("cfg", os.environ["SDC_CFG"]))
    if not 1 <= steps <= 50:
        raise ValueError("steps must be between 1 and 50")

    return {
        "prompt": prompt,
        "negative_prompt": job_input.get("negative_prompt", ""),
        "width": width,
        "height": height,
        "strength": float(job_input.get("strength", 1.0)),
        "seed": int(job_input.get("seed", -1)),
        "batch_count": 1,
        "auto_resize_ref_image": True,
        # Qwen Image Edit conditions on reference images (the documented
        # sd-cli invocation uses --ref-image).  Sending this as init_image
        # takes the generic IMG2IMG path and loses the Qwen edit conditioning.
        "init_image": None,
        "ref_images": [image_b64],
        "sample_params": {
            "scheduler": "discrete",
            "sample_method": "euler",
            "sample_steps": steps,
            "flow_shift": float(job_input.get("flow_shift", 3.1)),
            "guidance": {"txt_cfg": cfg, "img_cfg": None, "distilled_guidance": 0.0},
        },
        "lora": [{"path": os.environ["SDC_LORA"], "multiplier": 1.0, "is_high_noise": False}],
        # Qwen's VAE needs ~5 GB of temporary memory when decoding this image
        # in one pass.  Tile it by default so Q4 runs safely on a 24 GB GPU.
        "vae_tiling_params": {
            "enabled": bool(job_input.get("vae_tiling", True)),
            "temporal_tiling": False,
            "tile_size_x": 0,
            "tile_size_y": 0,
            "target_overlap": 0.5,
            "rel_size_x": 0.0,
            "rel_size_y": 0.0,
            "extra_tiling_args": "",
        },
        "output_format": "png",
        "output_compression": 100,
    }


def handler(job: dict[str, Any]) -> dict[str, Any]:
    try:
        with INFERENCE_LOCK:
            _start_server()
            payload = _request_payload(job["input"])
            started = time.monotonic()
            submission = requests.post(f"{BASE_URL}/sdcpp/v1/img_gen", json=payload, timeout=30)
            submission.raise_for_status()
            internal_job_id = submission.json()["id"]

            deadline = time.monotonic() + 15 * 60
            while time.monotonic() < deadline:
                status = requests.get(f"{BASE_URL}/sdcpp/v1/jobs/{internal_job_id}", timeout=30)
                status.raise_for_status()
                result = status.json()
                if result["status"] == "completed":
                    image = result["result"]["images"][0]["b64_json"]
                    elapsed = round(time.monotonic() - started, 3)
                    print(f"[sdcpp] completed in {elapsed}s", flush=True)
                    return {
                        "image": f"data:image/png;base64,{image}",
                        "seed": payload["seed"],
                        "width": payload["width"],
                        "height": payload["height"],
                        "steps": payload["sample_params"]["sample_steps"],
                        "elapsed_seconds": elapsed,
                        "runner": "stable-diffusion.cpp",
                    }
                if result["status"] in {"failed", "cancelled"}:
                    raise RuntimeError(json.dumps(result.get("error") or result))
                time.sleep(0.25)
            raise TimeoutError("generation exceeded 15 minutes")
    except Exception as exc:
        print(f"[sdcpp] inference error: {type(exc).__name__}: {exc}", flush=True)
        return {"error": f"{type(exc).__name__}: {exc}"}


print("[sdcpp] Runpod handler booted; model will load on first job", flush=True)
runpod.serverless.start({"handler": handler})
