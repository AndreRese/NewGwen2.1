#!/usr/bin/env python
"""Download the Qwen-Image 2.1 diffusion model + text encoder + VAE into ComfyUI/models.

    python download_models.py --model Q4_K_M            # default: uncensored Q4_K_M GGUF, 4.6 GB
    python download_models.py --model Q8_0 --text-encoder bf16
    python download_models.py --model BF16              # uncensored, unquantized (BF16 GGUF), 14.2 GB
    python download_models.py --model fp8               # uncensored fp8 safetensors (L4+), 7.1 GB
    python download_models.py --model base-bf16         # upstream base model, unquantized (Comfy-Org)
    python download_models.py --lora-only --lora https://huggingface.co/user/repo/blob/main/style.safetensors
    python download_models.py --lora-only --lora "https://civitai.com/api/download/models/123456" --lora-token KEY

Uncensored ("UC") files: main branch of abenzerps/Qwen-Image-2.1-Uncensored-GGUF — the
upstream weights with the author's fine-tune (LoRA, merged) on top. Plain upstream "base"
quants: the repo's `base` branch; unquantized base: Comfy-Org/Qwen-Image-2.1. The text
encoder and VAE are shared by both (abenzerps mirrors Comfy-Org's). No HF token needed.
Model names are case-insensitive. LoRAs (--lora URL, repeatable) go to ComfyUI/models/loras;
only .safetensors is accepted, and they must be Qwen-Image 2.1 LoRAs (1.x / 2.0 ones load
with "lora key not loaded" warnings and do nothing). Tokens: --lora-token, or the
CIVITAI_TOKEN / HF_TOKEN environment variables.
"""
import argparse
import importlib.metadata
import importlib.util
import os
import re
import struct
import sys
import urllib.parse

# fast downloads — must be set before huggingface_hub is imported
os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")  # huggingface_hub >= 1.0 (Xet backend)
if importlib.util.find_spec("hf_transfer") and importlib.metadata.version("huggingface_hub") < "1":
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")  # older hub versions

import requests  # noqa: E402
from huggingface_hub import hf_hub_download  # noqa: E402

REPO = "abenzerps/Qwen-Image-2.1-Uncensored-GGUF"
COMFY_ORG_REPO = "Comfy-Org/Qwen-Image-2.1"  # unquantized base + the optional w4a8 encoder

MODELS = {
    # name: (path in repo, repo, revision, approx size GB). *.gguf loads through
    # UnetLoaderGGUF, *.safetensors through the stock UNETLoader (generate.set_model).
    # The repo also has Q5_K_M / Q6_K; we only expose Q4_0 / Q4_K_M / Q8_0 + unquantized.
    # -- uncensored (main branch)
    "Q4_0": ("qwen-image-2.1-UC-Q4_0.gguf", REPO, "main", 4.15),
    "Q4_K_M": ("qwen-image-2.1-UC-Q4_K_M.gguf", REPO, "main", 4.60),
    "Q8_0": ("qwen-image-2.1-UC-Q8_0.gguf", REPO, "main", 7.59),
    "BF16": ("qwen-image-2.1-UC-BF16.gguf", REPO, "main", 14.23),
    "fp8": ("qwen-image-2.1-UC-fp8.safetensors", REPO, "main", 7.12),
    "int8": ("qwen-image-2.1-UC-int8_convrot.safetensors", REPO, "main", 7.26),
    # -- upstream base model
    "base-Q4_0": ("qwen-image-2.1-Q4_0.gguf", REPO, "base", 4.05),
    "base-Q4_K_M": ("qwen-image-2.1-Q4_K_M.gguf", REPO, "base", 4.60),
    "base-Q8_0": ("qwen-image-2.1-Q8_0.gguf", REPO, "base", 7.59),
    "base-int8": ("diffusion_models/qwen_image_2.1_int8_convrot.safetensors", COMFY_ORG_REPO, "main", 7.26),
    "base-bf16": ("diffusion_models/qwen_image_2.1_bf16.safetensors", COMFY_ORG_REPO, "main", 14.23),
}
TEXT_ENCODERS = {
    "int8": ("text_encoders/qwen3vl_8b_int8_convrot.safetensors", REPO, 9.35),
    "bf16": ("text_encoders/qwen3vl_8b_bf16.safetensors", REPO, 17.53),
    # smaller experimental encoder, lives only in the Comfy-Org repo
    "w4a8": ("text_encoders/qwen3vl_8b_w4a8.safetensors", COMFY_ORG_REPO, 6.3),
}
VAE = "vae/qwen_image_2.1_vae_bf16.safetensors"


def model_key(name):
    """Case-insensitive lookup: 'q4_k_m' -> 'Q4_K_M', 'bf16' -> 'BF16', 'BASE-BF16' -> 'base-bf16'."""
    for k in MODELS:
        if k.lower() == name.lower():
            return k
    raise argparse.ArgumentTypeError(f"unknown model {name!r}; choose from {', '.join(MODELS)}")


def diffusion_file(name):
    """File name as it appears in ComfyUI's model dropdowns, e.g. 'qwen-image-2.1-UC-Q4_K_M.gguf'."""
    return os.path.basename(MODELS[model_key(name)][0])


class LoraError(RuntimeError):
    pass


def _direct_url(url):
    """Hugging Face page links (…/blob/…) -> download links (…/resolve/…)."""
    return re.sub(r"^(https://huggingface\.co/[^/]+/[^/]+)/blob/", r"\1/resolve/", url.strip())


def _safe_name(name):
    name = os.path.basename(urllib.parse.unquote(name)).strip()
    return re.sub(r"[^\w.\-+()\[\] ]", "_", name)


def download_lora(url, comfy_dir="ComfyUI", name=None, token=None, progress=None):
    """Download a LoRA into ComfyUI/models/loras and return its file name (what the
    LoraLoaderModelOnly dropdown shows). Accepts Hugging Face file links (blob or resolve)
    and direct links such as https://civitai.com/api/download/models/<id>. Only real
    .safetensors files are kept (pickle formats can run code). progress(done, total) bytes."""
    url = _direct_url(url)
    if not url.startswith(("https://", "http://")):
        raise LoraError(f"not a URL: {url!r}")
    if token is None:
        token = os.environ.get("CIVITAI_TOKEN" if "civitai.com" in url else "HF_TOKEN")
    headers = {"Authorization": f"Bearer {token}"} if token else {}  # requests drops it on cross-host redirects
    lora_dir = os.path.join(comfy_dir, "models", "loras")
    os.makedirs(lora_dir, exist_ok=True)

    with requests.get(url, headers=headers, stream=True, timeout=60) as r:
        if r.status_code in (401, 403):
            raise LoraError(f"{r.status_code}: this link needs a token (Civitai API key / HF token)")
        if r.status_code != 200:
            raise LoraError(f"download failed: HTTP {r.status_code} for {url}")
        if "text/html" in r.headers.get("content-type", ""):
            raise LoraError("the link returned a web page, not a file — use the direct download link "
                            "(Civitai: the download button's URL; HF: the file's link)")
        if not name:
            m = re.search(r"filename\*?=(?:UTF-8'')?\"?([^\";]+)", r.headers.get("content-disposition", ""))
            name = m.group(1) if m else urllib.parse.urlparse(r.url).path
        name = _safe_name(name)
        if not name.lower().endswith(".safetensors"):
            raise LoraError(f"{name!r} is not a .safetensors file — only safetensors LoRAs are accepted")
        dst = os.path.join(lora_dir, name)
        total = int(r.headers.get("content-length") or 0)
        if os.path.isfile(dst) and total and os.path.getsize(dst) == total:
            print(f">> LoRA {name} already there")
            return name
        part, done = dst + ".part", 0
        with open(part, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                done += len(chunk)
                if progress:
                    progress(done, total)
    with open(part, "rb") as f:  # safetensors = u64 header length + JSON header
        head = f.read(9)
    if len(head) < 9 or head[8:9] != b"{" or struct.unpack("<Q", head[:8])[0] > done:
        os.remove(part)
        raise LoraError(f"{name!r} is not a valid safetensors file")
    os.replace(part, dst)
    print(f">> LoRA {name} ({done / 1e6:.1f} MB) -> {lora_dir}")
    return name


def fetch(repo, filename, local_dir, revision="main"):
    path = hf_hub_download(repo_id=repo, filename=filename, local_dir=local_dir, revision=revision)
    print(f"   {path}  ({os.path.getsize(path) / 1e9:.2f} GB)")
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--comfy-dir", default="ComfyUI", help="path to the ComfyUI checkout")
    ap.add_argument("--model", "--quant", dest="model", default="Q4_K_M", type=model_key, metavar="MODEL",
                    help=f"diffusion model, one of: {', '.join(MODELS)}")
    ap.add_argument("--text-encoder", default="int8", choices=sorted(TEXT_ENCODERS))
    ap.add_argument("--skip-text-encoder", action="store_true")
    ap.add_argument("--skip-vae", action="store_true")
    ap.add_argument("--lora", action="append", default=[], metavar="URL",
                    help="also download a LoRA (.safetensors) into models/loras; repeatable")
    ap.add_argument("--lora-token", help="Civitai API key / HF token for --lora links that need one")
    ap.add_argument("--lora-only", action="store_true", help="only download the --lora files")
    args = ap.parse_args()

    models = os.path.join(args.comfy_dir, "models")
    if not os.path.isdir(models):
        sys.exit(f"{models} not found — run setup_colab.sh first (or pass --comfy-dir)")

    for url in args.lora:
        try:
            download_lora(url, args.comfy_dir, token=args.lora_token)
        except (LoraError, requests.RequestException) as e:
            sys.exit(f">> LoRA download failed: {e}")
    if args.lora_only:
        return

    fname, repo, revision, size = MODELS[args.model]
    print(f">> diffusion model {os.path.basename(fname)} (~{size} GB, {repo}@{revision})")
    # abenzerps files sit at the repo root; the Comfy-Org ones already carry the diffusion_models/ prefix
    fetch(repo, fname, models if fname.startswith("diffusion_models/") else os.path.join(models, "diffusion_models"),
          revision)

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
    print(f"\n>> pick `{diffusion_file(args.model)}` in the Gradio / ComfyUI model dropdown")


if __name__ == "__main__":
    main()
