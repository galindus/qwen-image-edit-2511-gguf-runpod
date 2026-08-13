#!/usr/bin/env bash
# This file is deliberately copied into its own Docker build stage.  Do not
# combine it with application files: Docker will keep the expensive model layer
# cached until one of these model URLs or filenames is intentionally changed.
set -euo pipefail

MODEL_ROOT="${MODEL_ROOT:-/opt/models}"

download() {
    local url="$1"
    local destination="$2"
    mkdir -p "$(dirname "$destination")"
    # Hugging Face's public resolve URLs redirect to Xet/CDN.  curl follows the
    # redirect, retries transient CDN failures, and resumes a partial layer if
    # the build worker is interrupted.
    curl --fail --location --retry 8 --retry-all-errors --continue-at - \
        --output "$destination" "$url"
}

download \
  "https://huggingface.co/unsloth/Qwen-Image-Edit-2511-GGUF/resolve/main/qwen-image-edit-2511-Q4_K_M.gguf" \
  "$MODEL_ROOT/diffusion_models/qwen-image-edit-2511-Q4_K_M.gguf"
download \
  "https://huggingface.co/unsloth/Qwen2.5-VL-7B-Instruct-GGUF/resolve/main/Qwen2.5-VL-7B-Instruct-UD-Q4_K_XL.gguf" \
  "$MODEL_ROOT/text_encoders/Qwen2.5-VL-7B-Instruct-UD-Q4_K_XL.gguf"
download \
  "https://huggingface.co/unsloth/Qwen2.5-VL-7B-Instruct-GGUF/resolve/main/mmproj-BF16.gguf" \
  "$MODEL_ROOT/text_encoders/Qwen2.5-VL-7B-Instruct-mmproj-BF16.gguf"
download \
  "https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/vae/qwen_image_vae.safetensors" \
  "$MODEL_ROOT/vae/qwen_image_vae.safetensors"
download \
  "https://huggingface.co/lightx2v/Qwen-Image-Edit-2511-Lightning/resolve/main/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors" \
  "$MODEL_ROOT/loras/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors"
