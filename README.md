# Qwen Image Edit 2511 GGUF on Runpod Serverless

## Nunchaku INT4/FP4 runner (24 GB experiment)

`Dockerfile.nunchaku` is a separate direct-Diffusers endpoint for the community
Nunchaku build of Qwen Image Edit 2511 Lightning. It keeps an SVDQuant INT4
transformer and INT4 Qwen2.5-VL edit encoder resident in VRAM; it does not use
disk offload.

Create a separate Runpod endpoint with Dockerfile path `/Dockerfile.nunchaku`,
a 24 GB NVIDIA GPU and the usual network volume. The default precision detects
the GPU (`int4` on RTX 30/40 and most datacenter GPUs; `fp4` on Blackwell).
Override it only if required with `NUNCHAKU_PRECISION=int4` or `fp4`.
`NUNCHAKU_MEMORY_MODE=hybrid` is the default for a 24 GB card: it keeps the
INT4 transformer on GPU and offloads only the encoder/VAE between phases.
`model_cpu_offload` also offloads the transformer (safe but slower); `resident`
is intended only for GPUs with materially more free VRAM.

Pre-cache models on a temporary Pod with the same volume attached:

```bash
git clone https://github.com/galindus/qwen-image-edit-2511-gguf-runpod.git
cd qwen-image-edit-2511-gguf-runpod
pip install -U huggingface_hub
python3 download_nunchaku_models.py --cache-dir /runpod-volume/huggingface --precision int4
```

Request format:

```json
{"input":{"image":"data:image/jpeg;base64,...","prompt":"...","seed":42,"steps":4}}
```

This is an A/B experiment: compare the same ten inputs against the Q4 GGUF
endpoint before making it a production default.

## Alternative runner: stable-diffusion.cpp

`Dockerfile.sdcpp` is a second, independent Serverless image for an A/B
comparison against the ComfyUI endpoint. It runs Qwen through CUDA
`stable-diffusion.cpp`, loads the model once per worker, and exposes a compact
Runpod input contract instead of accepting a ComfyUI graph.

Create a **new** endpoint from this same Git repository, select
`/Dockerfile.sdcpp`, and leave the current Comfy endpoint unchanged. The image
includes the following files, so it requires **no network volume** and can be
deployed in any compatible Serverless region:

```text
models/diffusion_models/qwen-image-edit-2511-Q4_K_M.gguf
models/text_encoders/Qwen2.5-VL-7B-Instruct-UD-Q4_K_XL.gguf
models/text_encoders/Qwen2.5-VL-7B-Instruct-mmproj-BF16.gguf
models/vae/qwen_image_vae.safetensors
models/loras/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors
```

The ~20 GB model download is a dedicated Docker build stage. Changing
`handler_sdcpp.py` or other application configuration reuses that cached stage;
it downloads again only if `download_sdcpp_models.sh` (the pinned model set) is
changed, or if Runpod evicts its build cache.

Its default keeps the Q4 diffusion model on GPU and stages the Q4 Qwen2.5-VL
conditioner from host RAM to CUDA when needed, which fits a 24 GB worker. A
request is:

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

On a 24 GB GPU the default `SDC_MEMORY_MODE=clip_cpu` executes every graph on
`cuda0`, while setting `te=cpu,clip_vision=cpu` only in the parameter backend.
The Q4 transformer stays resident on GPU and Qwen2.5-VL weights are staged from
RAM only for the short conditioning phase, freeing roughly 6 GB during denoising
and VAE decode. `clip_vae_cpu` additionally stores VAE parameters in CPU RAM if
necessary; `resident` is for larger GPUs. `stream` is experimental and currently
not compatible with Qwen Edit 2511 graph cuts.

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
