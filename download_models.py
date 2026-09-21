#!/usr/bin/env python
"""Download the Qwen-Image 2.1 diffusion model (GGUF quant or original safetensors) +
text encoder + VAE into ComfyUI/models.

    python download_models.py --model Q4_K_M            # default, ~4.4 GB GGUF
    python download_models.py --model Q8_0 --text-encoder bf16
    python download_models.py --model bf16              # original unquantized weights, 13.3 GB
    python download_models.py --model int8              # ComfyUI-native int8 of the original, 6.8 GB

GGUFs come from abenzerps/Qwen-Image-2.1-Uncensored-GGUF (which also mirrors the text
encoder and VAE); the unquantized originals from Comfy-Org/Qwen-Image-2.1. No HF token.
"""
import argparse
import importlib.metadata
import importlib.util
import os
import sys

# fast downloads — must be set before huggingface_hub is imported
os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")  # huggingface_hub >= 1.0 (Xet backend)
if importlib.util.find_spec("hf_transfer") and importlib.metadata.version("huggingface_hub") < "1":
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")  # older hub versions

from huggingface_hub import hf_hub_download  # noqa: E402

REPO = "abenzerps/Qwen-Image-2.1-Uncensored-GGUF"
COMFY_ORG_REPO = "Comfy-Org/Qwen-Image-2.1"  # unquantized originals + the optional w4a8 encoder

MODELS = {
    # name: (filename in repo, repo, approx size GB). GGUF loads through UnetLoaderGGUF,
    # safetensors through the stock UNETLoader (see generate.py / build_workflows.py).
    # The GGUF repo also has Q5_K_M / Q6_K but we only expose these three.
    "Q4_0": ("qwen-image-2.1-Q4_0.gguf", REPO, 4.05),
    "Q4_K_M": ("qwen-image-2.1-Q4_K_M.gguf", REPO, 4.60),
    "Q8_0": ("qwen-image-2.1-Q8_0.gguf", REPO, 7.59),
    # originals (Comfy-Org repack of Qwen/Qwen-Image-2.1 — the same weights the GGUFs were made from)
    "int8": ("diffusion_models/qwen_image_2.1_int8_convrot.safetensors", COMFY_ORG_REPO, 7.26),
    "bf16": ("diffusion_models/qwen_image_2.1_bf16.safetensors", COMFY_ORG_REPO, 14.23),
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
    ap.add_argument("--model", "--quant", dest="model", default="Q4_K_M", choices=list(MODELS),
                    help="diffusion model: GGUF quant (Q4_0 / Q4_K_M / Q8_0) or original safetensors (int8 / bf16)")
    ap.add_argument("--text-encoder", default="int8", choices=sorted(TEXT_ENCODERS))
    ap.add_argument("--skip-text-encoder", action="store_true")
    ap.add_argument("--skip-vae", action="store_true")
    args = ap.parse_args()

    models = os.path.join(args.comfy_dir, "models")
    if not os.path.isdir(models):
        sys.exit(f"{models} not found — run setup_colab.sh first (or pass --comfy-dir)")

    fname, repo, size = MODELS[args.model]
    print(f">> diffusion model {os.path.basename(fname)} (~{size} GB)")
    # GGUFs sit at the repo root; the Comfy-Org files already carry the diffusion_models/ prefix
    fetch(repo, fname, models if fname.startswith("diffusion_models/") else os.path.join(models, "diffusion_models"))

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
