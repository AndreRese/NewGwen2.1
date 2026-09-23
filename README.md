# Qwen-Image 2.1 Uncensored (GGUF) — ComfyUI on Google Colab

ComfyUI setup for [abenzerps/Qwen-Image-2.1-Uncensored-GGUF](https://huggingface.co/abenzerps/Qwen-Image-2.1-Uncensored-GGUF):
the uncensored (`UC`) fine-tune of [Qwen/Qwen-Image-2.1](https://huggingface.co/Qwen/Qwen-Image-2.1)
(the author's LoRA merged into the upstream weights) as GGUF quants, an unquantized BF16 GGUF
and fp8 / int8 safetensors. The plain base model stays available (`--model base-*`).
Text-to-image (native 2K, N images per prompt), image edit via reference images, LoRAs.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AndreRese/NewGwen2.1/blob/main/Qwen_Image_2.1_GGUF_ComfyUI.ipynb)

Click the badge to open the notebook straight from this repo, or see **[COLAB.md](COLAB.md)** for the cell-by-cell guide.

## Layout

```
NewGwen2.1/
├── Qwen_Image_2.1_GGUF_ComfyUI.ipynb   # Colab notebook (runs the scripts below)
├── COLAB.md                            # same steps as prose
├── setup_colab.sh                      # ComfyUI + leejet/ComfyUI-GGUF at pinned commits + deps
├── download_models.py                  # diffusion model (UC / base, GGUF or safetensors) + text encoder + VAE + LoRAs by URL
├── launch_comfyui.py                   # start server, print Colab proxy / cloudflared URL
├── app_gradio.py                       # Gradio UI on *.gradio.live: t2i + edit tabs, N images, Send to Edit, LoRAs, host status
├── generate.py                         # headless t2i (run) / edit (edit) via the ComfyUI API; --count N, --model, --lora
├── colab_requirements.txt
├── packages.txt                        # apt packages (none needed at the moment)
└── workflows/
    ├── qwen_image_2.1_gguf_t2i.json          # text to image — drag into ComfyUI
    ├── qwen_image_2.1_gguf_t2i_api.json      # … API format, used by generate.run
    ├── qwen_image_2.1_gguf_edit.json         # image edit (LoadImage x2 → TextEncodeQwenImage21) — drag into ComfyUI
    ├── qwen_image_2.1_gguf_edit_api.json     # … API format, used by generate.edit
    ├── qwen_image_2.1_safetensors_t2i.json   # same graphs with the stock UNETLoader for the
    ├── qwen_image_2.1_safetensors_edit.json  # … fp8 / int8 / base-bf16 *.safetensors
    └── build_workflows.py                    # regenerates all of them (e.g. --quant Q8_0)
```

## Quick start (Colab)

```python
!git clone https://github.com/AndreRese/NewGwen2.1 qwen21
%cd qwen21
!bash setup_colab.sh
!python download_models.py --model Q4_K_M --text-encoder int8     # or --model BF16 for unquantized
!python app_gradio.py     # ComfyUI canvas (Colab proxy) + Gradio UI (*.gradio.live)
```

## Model files

| role | `--model` | file | size | source |
|---|---|---|---|---|
| diffusion, uncensored GGUF | `Q4_0` · **`Q4_K_M`** · `Q8_0` · `BF16` | `qwen-image-2.1-UC-{Q4_0,Q4_K_M,Q8_0,BF16}.gguf` | 4.2 / 4.6 / 7.6 / 14.2 GB | abenzerps `main` |
| diffusion, uncensored safetensors | `fp8` · `int8` | `qwen-image-2.1-UC-{fp8,int8_convrot}.safetensors` | 7.1 / 7.3 GB | abenzerps `main` |
| diffusion, base GGUF | `base-Q4_0` · `base-Q4_K_M` · `base-Q8_0` | `qwen-image-2.1-{Q4_0,Q4_K_M,Q8_0}.gguf` | 4.1 / 4.6 / 7.6 GB | abenzerps `base` branch |
| diffusion, base safetensors | `base-int8` · `base-bf16` | `qwen_image_2.1_{int8_convrot,bf16}.safetensors` | 7.3 / 14.2 GB | Comfy-Org |
| text encoder | `--text-encoder int8` / `bf16` / `w4a8` | `qwen3vl_8b_{int8_convrot,bf16,w4a8}.safetensors` | 9.4 / 17.5 / 6.3 GB | Comfy-Org (mirrored) |
| VAE | — | `qwen_image_2.1_vae_bf16.safetensors` | 0.7 GB | Comfy-Org (mirrored) |

`*.gguf` loads through `Unet Loader (GGUF)`, `*.safetensors` through the stock `Load Diffusion
Model` (`UNETLoader`); `generate.py` / the Gradio app pick the loader from the extension.
The repo also has Q5_K_M / Q6_K; they are left out of the menu on purpose.

## LoRAs

Only **Qwen-Image 2.1** LoRAs work (1.x / 2.0 ones load with `lora key not loaded` in the log and
change nothing). Put `.safetensors` files in `ComfyUI/models/loras`, or download them by URL:

```python
!python download_models.py --lora-only --lora https://huggingface.co/<user>/<repo>/blob/main/<file>.safetensors
!python download_models.py --lora-only --lora "https://civitai.com/api/download/models/<id>" --lora-token <API key>
```

The Gradio app has 3 LoRA slots (applied to both tabs) and the same download box;
`generate.py --lora file.safetensors:0.8` (repeatable) and `run(..., loras=[(name, strength)])`
chain `LoraLoaderModelOnly` nodes after the diffusion loader. The canvas workflows carry one
bypassed LoRA node (Ctrl+B to enable). Only `.safetensors` is accepted: `.ckpt` / `.pt` can run code.

## Pinned versions

`setup_colab.sh` checks out ComfyUI `b0f4b7b2` (2026-09-21, has the Qwen-Image 2.1 nodes) and
[leejet/ComfyUI-GGUF](https://github.com/leejet/ComfyUI-GGUF) `edd981b1`. The stock
`city96/ComfyUI-GGUF` does not know the Qwen-Image 2.1 architecture; the leejet fork (author of
stable-diffusion.cpp, which produced these quants) added it on 2026-09-20. `UPDATE_TO_LATEST=1
bash setup_colab.sh` uses ComfyUI master + the newest loader instead; bump the pins in the
script once a newer pair is tested.

## History: the Q8_0 norm-weight crash (fixed 2026-09-21)

The first `qwen-image-2.1-Q8_0.gguf` had its 1D `norm_q`/`norm_k` weights stored as Q8_0
blocks, which crashed `KSampler` with `Expected weight … shape [136] and normalized_shape = [128]`
([discussion #4](https://huggingface.co/abenzerps/Qwen-Image-2.1-Uncensored-GGUF/discussions/4)).
Fixed on both sides: the file was re-uploaded and the pinned loader dequantizes every quantized
1D tensor.

## Workflow

`UnetLoaderGGUF` (or the stock `UNETLoader` for `.safetensors`) → `KSampler` (euler / simple, 25 steps, cfg 1.0) with
`CLIPLoader(type=qwen_image)` → `TextEncodeQwenImage21` for conditioning,
`EmptyLatentImage` 1024² (or 2048² for native 2K) and `VAEDecode` → `SaveImage`.
This is Comfy-Org's official `image_qwen_image_2_1_t2i` template, flattened out of its
subgraph, with the loader swapped for GGUF.

**Image edit** (`image_qwen_image_2_1_image_edit` template, same treatment): `LoadImage` →
`TextEncodeQwenImage21` (`images.image_1…16` + VAE); `image_1` is the edit target and its
`latent` output sets the canvas; `QwenImage21Cache` (KV cache device / `int8` / `int4`)
between the loader and the sampler. Prompt with `<image1>`, `<image2>`, …
