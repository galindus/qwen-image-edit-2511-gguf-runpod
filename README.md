# Qwen Image Edit 2511 GGUF on Runpod Serverless

## Direct DiffSynth runner (recommended 24 GB memory experiment)

`Dockerfile.diffsynth` and `handler_diffsynth.py` provide a second, independent
Serverless worker that does **not** use ComfyUI or stable-diffusion.cpp. It uses
the official Qwen Edit 2511 model, the LightX2V 4-step Lightning LoRA and
DiffSynth's layer-level VRAM manager.

Select `/Dockerfile.diffsynth` when creating a new Runpod endpoint. Keep a
100 GB+ volume attached at `/runpod-volume`: the first job downloads roughly
58 GB of safetensors and future workers reuse that cache. Set `HF_TOKEN` as an
endpoint secret if Hugging Face access requires it.

The worker serializes jobs and after each result calls
`pipeline.load_models_to_device([])`, synchronizes CUDA and clears request
buffers. Its log reports free/allocated/reserved GPU memory after every job;
this is the check that matters for a 10-job queue.

Useful environment variables:

| Variable | Default | Meaning |
| --- | --- | --- |
| `QWEN_VRAM_LIMIT_GB` | GPU capacity minus 2 GB | Maximum model residency. Use `22` on a nominal 24 GB card. |
| `QWEN_STAGE_DTYPE` | `bf16` | `bf16` preserves quality; `fp8` reduces memory at a small quality cost. |
| `QWEN_MAX_OUTPUT_PIXELS` | `1048576` | Output cap while keeping original aspect ratio. |

Request payload:

```json
{"input":{"image":"data:image/jpeg;base64,...","prompt":"...","seed":42,"steps":4}}
```

For two references, send `images` instead of `image`. This is intentionally a
separate endpoint for comparison with the GGUF workers below.

## Alternative runner: stable-diffusion.cpp

`Dockerfile.sdcpp` is a second, independent Serverless image for an A/B
comparison against the ComfyUI endpoint. It runs Qwen through CUDA
`stable-diffusion.cpp`, loads the model once per worker, and exposes a compact
Runpod input contract instead of accepting a ComfyUI graph.

Create a **new** endpoint from this same Git repository, select
`/Dockerfile.sdcpp`, attach the existing network volume, and leave the current
Comfy endpoint unchanged. It requires the following extra files on that volume:

```text
models/diffusion_models/qwen-image-edit-2511-Q4_K_M.gguf
models/text_encoders/Qwen2.5-VL-7B-Instruct-UD-Q4_K_XL.gguf
models/text_encoders/Qwen2.5-VL-7B-Instruct-mmproj-BF16.gguf
models/vae/qwen_image_vae.safetensors
models/loras/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors
```

Its default is full-GPU Q4, with no CPU model offload. A request is:

```json
{
  "input": {
    "image": "data:image/jpeg;base64,...",
    "prompt": "Make the back room brighter while preserving the people.",
    "seed": 42,
    "steps": 4
  }
}
```

`image` may be a raw base64 string as well. The output defaults to the input
aspect ratio, capped to a 1536-pixel longest side; set `width` and `height`
(multiples of 16) to override it. The first request includes model loading;
compare only subsequent requests with ComfyUI.

For a closer A/B test with the Comfy workflow, set these endpoint environment
variables and redeploy. The FP8 encoder has no separate mmproj:

```text
SDC_TEXT_ENCODER=qwen_2.5_vl_7b_fp8_scaled.safetensors
SDC_MMPROJ=none
```

On a 24 GB GPU the default `SDC_MEMORY_MODE=stream` keeps the worker stable
across a queue by using the runner's CPU parameter streaming. Set
`SDC_MEMORY_MODE=resident` only when the complete Qwen Edit pipeline fits in
VRAM (for example, a larger GPU).

Runpod Serverless ComfyUI worker for Qwen Image Edit 2511 on a 24 GB GPU.

## Runtime

This image extends the official `runpod/worker-comfyui:5.8.6-base` worker and
installs `ComfyUI-GGUF`. It loads models from the network volume, so code-only
changes do not rebuild model layers.

The selected model is:

```text
unsloth/Qwen-Image-Edit-2511-GGUF/qwen-image-edit-2511-Q5_K_M.gguf
```

Q5_K_M has little spare VRAM after the Qwen2.5-VL FP8 text encoder and VAE load.
Start at 768×768 or equivalent. If the worker OOMs, change only the transformer
file to `qwen-image-edit-2511-Q4_K_M.gguf` and keep the rest of the setup.

## One-time volume setup

1. Create a 50 GB or larger Runpod network volume in the endpoint region.
2. Start a temporary GPU Pod with the volume mounted at `/workspace`.
3. Clone this repo and run:

   ```bash
   pip install -U huggingface_hub
   bash download_models.sh /workspace/models
   ```

4. Stop the Pod. The Serverless endpoint sees these files at
   `/runpod-volume/models/...` automatically.

## Deploy

1. In Runpod: **Serverless → New Endpoint → Import Git Repository**.
2. Use the `Dockerfile` in this repo and choose a 24 GB NVIDIA GPU.
3. Attach the prepared network volume under Advanced settings.
4. Select **Queue** and deploy. Enable `NETWORK_VOLUME_DEBUG=true` for the first
   boot if model discovery needs troubleshooting.

## Requests

This is the official Runpod ComfyUI Serverless API. Send a ComfyUI workflow
exported with **Workflow → Export (API)** plus optional base64 input images:

```json
{
  "input": {
    "workflow": { "...": "ComfyUI API workflow" },
    "images": [
      { "name": "source.png", "image": "data:image/png;base64,..." }
    ]
  }
}
```

Use Qwen's `TextEncodeQwenImageEditPlus` node, `UnetLoaderGGUF` for the Q5 model,
the Qwen2.5-VL encoder, and the Qwen Image VAE. The worker returns generated
images as base64 by default.
