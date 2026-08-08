# Qwen Image Edit 2511 GGUF on Runpod Serverless

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
