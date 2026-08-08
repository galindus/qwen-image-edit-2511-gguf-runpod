FROM runpod/worker-comfyui:5.8.6-base

# GGUF loader used by ComfyUI to load the quantized Qwen transformer.
RUN comfy-node-install https://github.com/city96/ComfyUI-GGUF.git

# Models deliberately live on the Serverless network volume at /runpod-volume.
# Do not bake the ~25 GB model set into the image: it makes rebuilds impractical.
