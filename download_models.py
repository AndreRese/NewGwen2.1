#!/usr/bin/env python
"""Download the Qwen-Image 2.1 Uncensored GGUF + companion files into ComfyUI/models.

    python download_models.py --quant Q4_K_M            # default, ~4.4 GB
    python download_models.py --quant Q8_0 --text-encoder bf16

Everything comes from a single repo (abenzerps/Qwen-Image-2.1-Uncensored-GGUF), which
mirrors the Comfy-Org text encoder and VAE, so no HF token is needed.
"""
import argparse
import os
import sys

# must be set before huggingface_hub is imported
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

from huggingface_hub import hf_hub_download  # noqa: E402

REPO = "abenzerps/Qwen-Image-2.1-Uncensored-GGUF"
COMFY_ORG_REPO = "Comfy-Org/Qwen-Image-2.1"  # only for the optional w4a8 encoder

QUANTS = {
    # name: approx size (GB). The repo also has Q5_K_M / Q6_K but we only expose these three.
    "Q4_0": 4.05,
    "Q4_K_M": 4.60,
    "Q8_0": 7.59,
}
TEXT_ENCODERS = {
    "int8": ("text_encoders/qwen3vl_8b_int8_convrot.safetensors", REPO, 9.35),
    "bf16": ("text_encoders/qwen3vl_8b_bf16.safetensors", REPO, 17.53),
    # smaller experimental encoder, lives only in the Comfy-Org repo
    "w4a8": ("text_encoders/qwen3vl_8b_w4a8.safetensors", COMFY_ORG_REPO, 6.3),
}
VAE = "vae/qwen_image_2.1_vae_bf16.safetensors"


def fetch(repo, filename, local_dir):
    path = hf_hub_download(repo_id=repo, filename=filename, local_dir=local_dir)
    print(f"   {path}  ({os.path.getsize(path) / 2**30:.2f} GB)")
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--comfy-dir", default="ComfyUI", help="path to the ComfyUI checkout")
    ap.add_argument("--quant", default="Q4_K_M", choices=sorted(QUANTS), help="GGUF quant of the diffusion model")
    ap.add_argument("--text-encoder", default="int8", choices=sorted(TEXT_ENCODERS))
    ap.add_argument("--skip-text-encoder", action="store_true")
    ap.add_argument("--skip-vae", action="store_true")
    args = ap.parse_args()

    models = os.path.join(args.comfy_dir, "models")
    if not os.path.isdir(models):
        sys.exit(f"{models} not found — run setup_colab.sh first (or pass --comfy-dir)")

    gguf = f"qwen-image-2.1-{args.quant}.gguf"
    print(f">> diffusion model {gguf} (~{QUANTS[args.quant]} GB)")
    fetch(REPO, gguf, os.path.join(models, "diffusion_models"))

    if not args.skip_text_encoder:
        fname, repo, size = TEXT_ENCODERS[args.text_encoder]
        print(f">> text encoder {os.path.basename(fname)} (~{size} GB)")
        fetch(repo, fname, models)  # repo path already starts with text_encoders/

    if not args.skip_vae:
        print(f">> VAE {os.path.basename(VAE)} (~0.68 GB)")
        fetch(REPO, VAE, models)

    print("\nLayout:")
    for sub in ("diffusion_models", "text_encoders", "vae"):
        d = os.path.join(models, sub)
        for f in sorted(os.listdir(d)) if os.path.isdir(d) else []:
            if f.endswith((".gguf", ".safetensors")):
                print(f"  models/{sub}/{f}")


if __name__ == "__main__":
    main()
