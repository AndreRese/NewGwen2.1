# Qwen-Image 2.1 Uncensored (GGUF) — ComfyUI on Google Colab

ComfyUI setup for [abenzerps/Qwen-Image-2.1-Uncensored-GGUF](https://huggingface.co/abenzerps/Qwen-Image-2.1-Uncensored-GGUF),
GGUF quants of [Qwen/Qwen-Image-2.1](https://huggingface.co/Qwen/Qwen-Image-2.1) (built from the
upstream weights; the repo says a fully uncensored build is still in the works), plus the
**original unquantized weights** from
[Comfy-Org/Qwen-Image-2.1](https://huggingface.co/Comfy-Org/Qwen-Image-2.1) (`--model bf16` / `int8`).
Text-to-image (native 2K, N images per prompt) and image edit via reference images.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AndreRese/NewGwen2.1/blob/main/Qwen_Image_2.1_GGUF_ComfyUI.ipynb)

Click the badge to open the notebook straight from this repo, or see **[COLAB.md](COLAB.md)** for the cell-by-cell guide.

## Layout

```
NewGwen2.1/
├── Qwen_Image_2.1_GGUF_ComfyUI.ipynb   # Colab notebook (runs the scripts below)
├── COLAB.md                            # same steps as prose
├── setup_colab.sh                      # ComfyUI master + leejet/ComfyUI-GGUF + deps
├── download_models.py                  # diffusion model (GGUF quant or original safetensors) + text encoder + VAE
├── launch_comfyui.py                   # start server, print Colab proxy / cloudflared URL
├── app_gradio.py                       # Gradio UI on *.gradio.live: Text to Image + Image Edit tabs, N images, host status
├── generate.py                         # headless t2i (run) / edit (edit) via the ComfyUI API; --count N, --model
├── colab_requirements.txt
├── packages.txt                        # apt packages
└── workflows/
    ├── qwen_image_2.1_gguf_t2i.json       # text to image — drag into ComfyUI
    ├── qwen_image_2.1_gguf_t2i_api.json   # … API format, used by generate.run
    ├── qwen_image_2.1_gguf_edit.json      # image edit (LoadImage x2 → TextEncodeQwenImage21) — drag into ComfyUI
    ├── qwen_image_2.1_gguf_edit_api.json  # … API format, used by generate.edit
    ├── qwen_image_2.1_bf16_t2i.json       # same graphs with the stock UNETLoader for the
    ├── qwen_image_2.1_bf16_edit.json      # … original *.safetensors
    └── build_workflows.py                 # regenerates all of them (e.g. --quant Q8_0)
```

## Quick start (Colab)

```python
!git clone https://github.com/AndreRese/NewGwen2.1 qwen21
%cd qwen21
!bash setup_colab.sh
!python download_models.py --model Q4_K_M --text-encoder int8     # or --model bf16 for the original
!python app_gradio.py     # ComfyUI canvas (Colab proxy) + Gradio UI (*.gradio.live)
```

## Model files

| role | file | size | source |
|---|---|---|---|
| diffusion model (GGUF) | `qwen-image-2.1-{Q4_0,Q4_K_M,Q8_0}.gguf` (pick one) | 4.0 / 4.6 / 7.6 GB | abenzerps |
| diffusion model (original) | `qwen_image_2.1_{int8_convrot,bf16}.safetensors` — unquantized upstream weights, stock `UNETLoader` | 7.3 / 14.2 GB | Comfy-Org |
| text encoder | `qwen3vl_8b_int8_convrot.safetensors` / `qwen3vl_8b_bf16.safetensors` | 9.4 / 17.5 GB | mirrored from Comfy-Org/Qwen-Image-2.1 |
| VAE | `qwen_image_2.1_vae_bf16.safetensors` | 0.7 GB | mirrored from Comfy-Org/Qwen-Image-2.1 |

## Why the leejet fork

The stock `city96/ComfyUI-GGUF` does not know the Qwen-Image 2.1 architecture. The
[leejet fork](https://github.com/leejet/ComfyUI-GGUF) (author of stable-diffusion.cpp, which
produced these quants) added it on 2026-09-20. `setup_colab.sh` pins that fork; switch back
to upstream once it merges.

## Note: the Q8_0 norm-weight crash (fixed 2026-09-21)

The first `qwen-image-2.1-Q8_0.gguf` had its 1D `norm_q`/`norm_k` weights stored as Q8_0
blocks, which crashed `KSampler` with `Expected weight … shape [136] and normalized_shape = [128]`
([discussion #4](https://huggingface.co/abenzerps/Qwen-Image-2.1-Uncensored-GGUF/discussions/4)).
Both sides fixed it on 2026-09-21: the HF repo re-uploaded Q8_0 and the leejet loader now
dequantizes every quantized 1D tensor (`edd981b1`). If you downloaded Q8_0 before that, delete
`ComfyUI/models/diffusion_models/qwen-image-2.1-Q8_0.gguf` and run the download cell again.

## Workflow

`UnetLoaderGGUF` (or the stock `UNETLoader` for the `.safetensors` originals) → `KSampler` (euler / simple, 25 steps, cfg 1.0) with
`CLIPLoader(type=qwen_image)` → `TextEncodeQwenImage21` for conditioning,
`EmptyLatentImage` 1024² (or 2048² for native 2K) and `VAEDecode` → `SaveImage`.
This is Comfy-Org's official `image_qwen_image_2_1_t2i` template, flattened out of its
subgraph, with the loader swapped for GGUF.

**Image edit** (`image_qwen_image_2_1_image_edit` template, same treatment): `LoadImage` →
`TextEncodeQwenImage21` (`images.image_1…16` + VAE); `image_1` is the edit target and its
`latent` output sets the canvas; `QwenImage21Cache` (KV cache device / `int8` / `int4`)
between the GGUF loader and the sampler. Prompt with `<image1>`, `<image2>`, …
