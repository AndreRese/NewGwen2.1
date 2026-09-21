#!/usr/bin/env python
"""Make ComfyUI-GGUF dequantize *any* quantized 1D tensor (norm weights / biases) at load time.

    python patch_gguf_loader.py [--comfy-dir ComfyUI]

Why: abenzerps' qwen-image-2.1-Q8_0.gguf was converted with the 1D norm_q / norm_k
weights quantized to Q8_0 blocks (128 values -> 136 bytes). ComfyUI's Qwen-Image 2.1
model passes `self.norm_q.weight` straight into `rms_rope`, bypassing the GGUF ops layer,
so the raw block bytes reach torch.rms_norm:

    RuntimeError: Expected weight to be of same shape as normalized_shape,
                  but got weight of shape [136] and normalized_shape = [128]

The loader already dequantizes 1D BF16 tensors to float32 ("1D tensors shouldn't be
quantized"); this widens that rule to every non-F32/F16 type. Idempotent — safe to re-run.
Q4_K_M / Q5_K_M / Q6_K keep norms in BF16 and never needed this.
"""
import argparse
import os
import sys

OLD = "if len(shape) <= 1 and tensor.tensor_type == gguf.GGMLQuantizationType.BF16:"
NEW = ("if len(shape) <= 1 and tensor.tensor_type not in "
       "{gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16}:  # patched: any quantized 1D")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--comfy-dir", default="ComfyUI")
    a = ap.parse_args()
    path = os.path.join(a.comfy_dir, "custom_nodes", "ComfyUI-GGUF", "loader.py")
    if not os.path.isfile(path):
        sys.exit(f"{path} not found")
    src = open(path, encoding="utf-8").read()
    if NEW in src:
        print(f">> {path}: already patched")
        return
    if OLD not in src:
        sys.exit(f">> {path}: expected line not found — loader changed upstream, check manually:\n   {OLD}")
    open(path, "w", encoding="utf-8").write(src.replace(OLD, NEW))
    print(f">> {path}: patched (quantized 1D tensors are now dequantized to fp32 at load)")


if __name__ == "__main__":
    main()
