#!/usr/bin/env bash
# One-shot environment setup for Google Colab.
#   bash setup_colab.sh [COMFY_DIR]
# Clones ComfyUI (master — TextEncodeQwenImage21 is only on master, not in a tagged
# release yet) and the leejet fork of ComfyUI-GGUF, which is the only GGUF loader with
# Qwen-Image 2.1 support as of 2026-09-20 (city96 upstream has not merged it).
set -euo pipefail

COMFY_DIR="${1:-ComfyUI}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo ">> apt packages"
if [ -f "$HERE/packages.txt" ]; then
  xargs -a "$HERE/packages.txt" apt-get -qq install -y >/dev/null
fi

echo ">> ComfyUI -> $COMFY_DIR"
if [ ! -d "$COMFY_DIR/.git" ]; then
  git clone -q --depth 1 https://github.com/comfyanonymous/ComfyUI "$COMFY_DIR"
else
  git -C "$COMFY_DIR" pull -q --ff-only
fi

echo ">> ComfyUI-GGUF (leejet fork, Qwen-Image 2.1 support)"
GGUF_DIR="$COMFY_DIR/custom_nodes/ComfyUI-GGUF"
if [ ! -d "$GGUF_DIR/.git" ]; then
  git clone -q --depth 1 https://github.com/leejet/ComfyUI-GGUF "$GGUF_DIR"
else
  git -C "$GGUF_DIR" pull -q --ff-only
fi
# Q8_0 has quantized 1D norm weights -> make the loader dequantize them (see patch_gguf_loader.py)
python "$HERE/patch_gguf_loader.py" --comfy-dir "$COMFY_DIR"

echo ">> python deps"
pip install -q -r "$COMFY_DIR/requirements.txt"
pip install -q -r "$GGUF_DIR/requirements.txt"
pip install -q -r "$HERE/colab_requirements.txt"

echo ">> copy workflows into ComfyUI/user/default/workflows"
mkdir -p "$COMFY_DIR/user/default/workflows"
cp "$HERE"/workflows/*_t2i.json "$COMFY_DIR/user/default/workflows/" 2>/dev/null || true

python - <<'PY'
import torch
print(f"torch {torch.__version__} cuda {torch.version.cuda} "
      f"gpu {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE'}")
PY
echo ">> done. Next: python download_models.py --quant Q4_K_M"
