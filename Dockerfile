FROM runpod/worker-comfyui:5.8.6-base

# Bump this marker to explicitly trigger a GitHub-based Runpod rebuild.
LABEL org.opencontainers.image.revision="qwen-gguf-20260809-normalvram"

# GGUF loader used by ComfyUI to load the quantized Qwen transformer.
# `comfy-node-install` only resolves registry package names; clone this
# GitHub-only node explicitly so the loader is present at worker startup.
RUN git clone --depth 1 https://github.com/city96/ComfyUI-GGUF.git \
      /comfyui/custom_nodes/ComfyUI-GGUF \
 && python -m pip install --no-cache-dir -r /comfyui/custom_nodes/ComfyUI-GGUF/requirements.txt

# Keep the worker's default VRAM policy, but allow an endpoint to override it
# without rebuilding the image (for example: COMFYUI_EXTRA_ARGS=--normalvram).
ENV COMFYUI_EXTRA_ARGS=""
RUN sed -i 's/--log-stdout \&/${COMFYUI_EXTRA_ARGS:-} --log-stdout \&/' /start.sh

# Models deliberately live on the Serverless network volume at /runpod-volume.
# Do not bake the ~25 GB model set into the image: it makes rebuilds impractical.
