#!/usr/bin/env python3
"""Pre-cache exactly the files needed by the Nunchaku 2511 worker.

Run this in a temporary Runpod Pod with its network volume mounted. It avoids
the upstream BF16 transformer because the endpoint supplies SVDQuant weights.
"""

from __future__ import annotations

import argparse

from huggingface_hub import snapshot_download

MODEL_REPO = "tonera/Qwen-Image-Edit-2511-Lightning-Nunchaku"
TEXT_ENCODER_REPO = "tonera/Qwen2.5vl-Nunchaku"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", default="/runpod-volume/huggingface")
    parser.add_argument("--precision", choices=("int4", "fp4"), default="int4")
    args = parser.parse_args()

    snapshot_download(
        MODEL_REPO,
        cache_dir=args.cache_dir,
        allow_patterns=[
            "model_index.json",
            "scheduler/*",
            "tokenizer/*",
            "tokenizer_2/*",
            "text_encoder_2/*",
            "vae/*",
            "image_processor/*",
            "processor/*",
            "*.json",
            f"svdq-{args.precision}_r32-Qwen-Image-Edit-2511-Lightning-Nunchaku.safetensors",
        ],
    )
    snapshot_download(
        TEXT_ENCODER_REPO,
        cache_dir=args.cache_dir,
        allow_patterns=["svdq-int4-Qwen2.5vl-Nunchaku.safetensors", "*.json"],
    )
    print(f"Nunchaku {args.precision} cache ready in {args.cache_dir}")


if __name__ == "__main__":
    main()
