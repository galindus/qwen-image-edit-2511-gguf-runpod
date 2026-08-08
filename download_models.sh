#!/usr/bin/env bash
set -euo pipefail

# Run this once on a temporary Runpod Pod with the same network volume mounted
# at /workspace. It downloads the model set used by the Serverless endpoint.
MODEL_ROOT="${1:-/workspace/models}"

mkdir -p "$MODEL_ROOT/diffusion_models" "$MODEL_ROOT/text_encoders" "$MODEL_ROOT/vae"

huggingface-cli download unsloth/Qwen-Image-Edit-2511-GGUF \
  qwen-image-edit-2511-Q5_K_M.gguf \
  --local-dir "$MODEL_ROOT/diffusion_models"

# The official ComfyUI Qwen workflow uses this FP8-scaled Qwen2.5-VL encoder.
huggingface-cli download Comfy-Org/HunyuanVideo_1.5_repackaged \
  split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors \
  --local-dir "$MODEL_ROOT/text_encoders" \
  --local-dir-use-symlinks False
mv "$MODEL_ROOT/text_encoders/split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors" \
  "$MODEL_ROOT/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors"
rmdir "$MODEL_ROOT/text_encoders/split_files/text_encoders" "$MODEL_ROOT/text_encoders/split_files" || true

huggingface-cli download Comfy-Org/Qwen-Image_ComfyUI \
  split_files/vae/qwen_image_vae.safetensors \
  --local-dir "$MODEL_ROOT/vae" \
  --local-dir-use-symlinks False
mv "$MODEL_ROOT/vae/split_files/vae/qwen_image_vae.safetensors" \
  "$MODEL_ROOT/vae/qwen_image_vae.safetensors"
rmdir "$MODEL_ROOT/vae/split_files/vae" "$MODEL_ROOT/vae/split_files" || true
