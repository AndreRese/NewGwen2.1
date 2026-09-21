# Qwen-Image 2.1 Uncensored (GGUF) — ComfyUI on Google Colab

ComfyUI setup for [abenzerps/Qwen-Image-2.1-Uncensored-GGUF](https://huggingface.co/abenzerps/Qwen-Image-2.1-Uncensored-GGUF),
a GGUF quant of [Qwen/Qwen-Image-2.1](https://huggingface.co/Qwen/Qwen-Image-2.1) with no
safety checker. Text-to-image (native 2K) and image edit via reference images.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AndreRese/NewGwen2.1/blob/main/Qwen_Image_2.1_GGUF_ComfyUI.ipynb)

Click the badge to open the notebook straight from this repo, or see **[COLAB.md](COLAB.md)** for the cell-by-cell guide.

## Layout

```
NewGwen2.1/
├── Qwen_Image_2.1_GGUF_ComfyUI.ipynb   # Colab notebook (runs the scripts below)
├── COLAB.md                            # same steps as prose
├── setup_colab.sh                      # ComfyUI master + leejet/ComfyUI-GGUF + deps
├── download_models.py                  # GGUF (pick quant) + text encoder + VAE -> ComfyUI/models
├── launch_comfyui.py                   # start server, print Colab proxy / cloudflared URL
├── generate.py                         # headless t2i through the ComfyUI API
├── colab_requirements.txt
├── packages.txt                        # apt packages
└── workflows/
    ├── qwen_image_2.1_gguf_t2i.json       # drag into ComfyUI
    ├── qwen_image_2.1_gguf_t2i_api.json   # used by generate.py
    └── build_workflows.py                 # regenerates both (e.g. --quant Q8_0)
```

## Quick start (Colab)

```python
!git clone https://github.com/AndreRese/NewGwen2.1 qwen21
%cd qwen21
!bash setup_colab.sh
!python download_models.py --quant Q4_K_M --text-encoder int8
from launch_comfyui import launch; launch()
```

## Model files

| role | file | size | source |
|---|---|---|---|
| diffusion model | `qwen-image-2.1-{Q4_0,Q4_K_M,Q8_0}.gguf` (pick one) | 4.0 / 4.6 / 7.6 GB | abenzerps (uncensored) |
| text encoder | `qwen3vl_8b_int8_convrot.safetensors` / `qwen3vl_8b_bf16.safetensors` | 9.4 / 17.5 GB | mirrored from Comfy-Org/Qwen-Image-2.1 |
| VAE | `qwen_image_2.1_vae_bf16.safetensors` | 0.7 GB | mirrored from Comfy-Org/Qwen-Image-2.1 |

## Why the leejet fork

The stock `city96/ComfyUI-GGUF` does not know the Qwen-Image 2.1 architecture. The
[leejet fork](https://github.com/leejet/ComfyUI-GGUF) (author of stable-diffusion.cpp, which
produced these quants) added it on 2026-09-20. `setup_colab.sh` pins that fork; switch back
to upstream once it merges.

## Workflow

`UnetLoaderGGUF` → `KSampler` (euler / simple, 25 steps, cfg 1.0) with
`CLIPLoader(type=qwen_image)` → `TextEncodeQwenImage21` for conditioning,
`EmptyLatentImage` 1024² (or 2048² for native 2K) and `VAEDecode` → `SaveImage`.
This is Comfy-Org's official `image_qwen_image_2_1_t2i` template, flattened out of its
subgraph, with the loader swapped for GGUF.
