#!/usr/bin/env bash
# One-shot environment setup for Google Colab.
#   bash setup_colab.sh [COMFY_DIR]
#   UPDATE_TO_LATEST=1 bash setup_colab.sh     # ComfyUI master + newest loader instead of the pins
#
# ComfyUI and the leejet fork of ComfyUI-GGUF (the only GGUF loader with Qwen-Image 2.1
# support; city96 upstream has not merged it) are pinned to commits known to work together.
# Unpinned master already broke this setup once (2026-09-21). Bump the pins after testing.
set -euo pipefail

COMFY_DIR="${1:-ComfyUI}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

COMFY_URL=https://github.com/comfyanonymous/ComfyUI
COMFY_PIN=b0f4b7b294ce482a2e071d9d762c133d38c7aa07   # master 2026-09-21 (after v0.37.0, Qwen-Image 2.1 nodes)
GGUF_URL=https://github.com/leejet/ComfyUI-GGUF
GGUF_PIN=edd981b10e107d3b8f58e16c498f2d08f631bc47    # 2026-09-21 "dequantize every quantized 1D tensor"

if [ "${UPDATE_TO_LATEST:-0}" = "1" ]; then
  COMFY_REF=master GGUF_REF=main
  echo ">> UPDATE_TO_LATEST=1: using ComfyUI master + ComfyUI-GGUF main (untested combination)"
else
  COMFY_REF=$COMFY_PIN GGUF_REF=$GGUF_PIN
fi

# shallow checkout of a branch or commit; works on a fresh dir and on an existing clone
checkout() {  # dir url ref
  local dir=$1 url=$2 ref=$3
  if [ ! -d "$dir/.git" ]; then
    mkdir -p "$dir"
    git -C "$dir" init -q
    git -C "$dir" remote add origin "$url"
  fi
  git -C "$dir" fetch -q --depth 1 origin "$ref"
  git -C "$dir" checkout -q --force FETCH_HEAD
  echo "   $(basename "$dir") @ $(git -C "$dir" rev-parse --short HEAD)"
}

echo ">> apt packages"
PKGS=$(grep -v '^\s*#' "$HERE/packages.txt" 2>/dev/null | xargs || true)
if [ -n "$PKGS" ]; then
  # never fatal: a stale apt index must not abort the whole setup
  (apt-get -qq update && apt-get -qq install -y $PKGS) >/dev/null 2>&1 || echo "   warning: apt install failed ($PKGS)"
else
  echo "   none needed"
fi

echo ">> ComfyUI -> $COMFY_DIR"
checkout "$COMFY_DIR" "$COMFY_URL" "$COMFY_REF"

echo ">> ComfyUI-GGUF (leejet fork, Qwen-Image 2.1 support)"
GGUF_DIR="$COMFY_DIR/custom_nodes/ComfyUI-GGUF"
checkout "$GGUF_DIR" "$GGUF_URL" "$GGUF_REF"

echo ">> python deps"
pip install -q -r "$COMFY_DIR/requirements.txt"
pip install -q -r "$GGUF_DIR/requirements.txt"
pip install -q -r "$HERE/colab_requirements.txt"

echo ">> copy workflows into ComfyUI/user/default/workflows"
mkdir -p "$COMFY_DIR/user/default/workflows"
cp "$HERE"/workflows/*_t2i.json "$HERE"/workflows/*_edit.json "$COMFY_DIR/user/default/workflows/" 2>/dev/null || true

python - <<'PY'
import torch
print(f"torch {torch.__version__} cuda {torch.version.cuda} "
      f"gpu {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE'}")
PY
echo ">> done. Next: python download_models.py --model Q4_K_M"
