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
    # The Hugging Face public URL redirects to Xet/CDN. aria2 uses 16 ranged
    # connections to that CDN instead of downloading one 13 GB file serially.
    # This keeps each Docker layer well below Runpod's 30-minute build limit.
    aria2c --allow-overwrite=true --auto-file-renaming=false --continue=true \
        --max-connection-per-server=16 --split=16 --min-split-size=8M \
        --retry-wait=2 --max-tries=8 --file-allocation=none \
        --dir "$(dirname "$destination")" --out "$(basename "$destination")" \
        "$url"
}

case "${1:?usage: download_sdcpp_models <transformer|encoder|mmproj|vae|lora>}" in
  transformer)
    download "https://huggingface.co/unsloth/Qwen-Image-Edit-2511-GGUF/resolve/main/qwen-image-edit-2511-Q4_K_M.gguf" "$MODEL_ROOT/diffusion_models/qwen-image-edit-2511-Q4_K_M.gguf" ;;
  encoder)
    download "https://huggingface.co/unsloth/Qwen2.5-VL-7B-Instruct-GGUF/resolve/main/Qwen2.5-VL-7B-Instruct-UD-Q4_K_XL.gguf" "$MODEL_ROOT/text_encoders/Qwen2.5-VL-7B-Instruct-UD-Q4_K_XL.gguf" ;;
  mmproj)
    download "https://huggingface.co/unsloth/Qwen2.5-VL-7B-Instruct-GGUF/resolve/main/mmproj-BF16.gguf" "$MODEL_ROOT/text_encoders/Qwen2.5-VL-7B-Instruct-mmproj-BF16.gguf" ;;
  vae)
    download "https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/vae/qwen_image_vae.safetensors" "$MODEL_ROOT/vae/qwen_image_vae.safetensors" ;;
  lora)
    download "https://huggingface.co/lightx2v/Qwen-Image-Edit-2511-Lightning/resolve/main/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors" "$MODEL_ROOT/loras/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors" ;;
  *) echo "unknown model: $1" >&2; exit 2 ;;
esac
