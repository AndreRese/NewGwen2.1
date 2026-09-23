# Running on Google Colab (ComfyUI)

Runs **Qwen-Image 2.1 Uncensored** (GGUF / safetensors) inside ComfyUI on a Colab GPU. Same
idea as the Wan2.2 project's `COLAB.md`, but ComfyUI is the app: no Gradio port, you get the
ComfyUI web UI through Colab's port proxy (or a cloudflared tunnel), a Gradio front-end on a
public `*.gradio.live` link, plus `generate.py` for headless runs from a cell.

The whole thing is also packaged as a notebook: `Qwen_Image_2.1_GGUF_ComfyUI.ipynb`.

**Runtime → Change runtime type → GPU.** Check with `!nvidia-smi`.

| GPU | VRAM / RAM | Recommended | Notes |
|---|---|---|---|
| T4 (free) | 15 GB / 12.7 GB | `Q4_K_M` + `int8` encoder, `--lowvram` | Works but tight — the 9 GB text encoder is mmap'd from disk and offloaded after encoding. Slow; no bf16 / fp8 on sm75, so stay on Q4/Q8 GGUFs. |
| L4 | 22.5 GB / 53 GB | `Q4_K_M`…`Q8_0`, or `fp8` / `int8` safetensors, + `int8` encoder | Sweet spot. Everything stays resident. |
| A100 40/80 GB | 40–80 GB / 83 GB | `BF16` (unquantized) + `bf16` encoder | Fastest and the quality ceiling. |

## Cell 1 – get the code

```python
!git clone https://github.com/AndreRese/NewGwen2.1 qwen21
%cd qwen21
```

(or upload the folder: `setup_colab.sh`, `download_models.py`, `launch_comfyui.py`,
`generate.py`, `app_gradio.py`, `colab_requirements.txt`, `packages.txt` and `workflows/`)

## Cell 2 – ComfyUI + custom nodes + dependencies

```python
!bash setup_colab.sh
# !UPDATE_TO_LATEST=1 bash setup_colab.sh    # ComfyUI master + newest loader (untested combination)
```

Checks out ComfyUI and the **leejet fork** of ComfyUI-GGUF (the only GGUF loader with
Qwen-Image 2.1 support; city96 upstream hasn't merged it) at **pinned commits** known to work
together — re-running the cell resets both to those commits. Installs both
`requirements.txt` files plus `colab_requirements.txt`, and copies the workflows into
`ComfyUI/user/default/workflows/` so they show up in the UI's workflow browser.

## Cell 3 – models (~14 GB for the default combo)

```python
!python download_models.py --model Q4_K_M --text-encoder int8
!python download_models.py --model BF16          # uncensored, unquantized (14.2 GB)
!python download_models.py --model base-Q4_K_M   # plain upstream base model instead
```

No HF token needed. Uncensored (`UC`) files come from the main branch of
`abenzerps/Qwen-Image-2.1-Uncensored-GGUF`, base quants from its `base` branch, the
unquantized base from `Comfy-Org/Qwen-Image-2.1`; the text encoder and VAE are shared.
Only the selected model is downloaded. Options (case-insensitive):

| flag | choices | |
|---|---|---|
| `--model` | uncensored: `Q4_0` 4.15 GB · **`Q4_K_M` 4.60 GB** · `Q8_0` 7.59 GB · `BF16` 14.2 GB (GGUF) · `fp8` 7.12 GB · `int8` 7.26 GB (safetensors) — base: `base-Q4_0` · `base-Q4_K_M` · `base-Q8_0` (GGUF) · `base-int8` · `base-bf16` (safetensors). `--quant` still works as an alias. | diffusion model |
| `--text-encoder` | **`int8`** 9.35 GB · `bf16` 17.5 GB · `w4a8` 6.3 GB (from Comfy-Org, experimental) | Qwen3-VL 8B |

Files land in:

```
ComfyUI/models/
├── diffusion_models/qwen-image-2.1-UC-Q4_K_M.gguf
├── text_encoders/qwen3vl_8b_int8_convrot.safetensors
└── vae/qwen_image_2.1_vae_bf16.safetensors
```

Re-running the cell checks the files against the repo and re-downloads one only if it changed
upstream.

## Cell 4 – launch ComfyUI + Gradio

```python
!python app_gradio.py              # L4 / A100
# !python app_gradio.py --lowvram  # T4
# !python app_gradio.py --tunnel   # also a *.trycloudflare.com URL for the canvas
```

Starts ComfyUI in the background, then a Gradio app with `share=True`. You get:

- **ComfyUI canvas** — `https://….colab.googleusercontent.com` (Colab's port proxy; only
  works while you're logged into that Colab session). **Workflow → Open** →
  `qwen_image_2.1_gguf_t2i` / `_edit` for GGUF files, `qwen_image_2.1_safetensors_t2i` / `_edit`
  for `.safetensors` (stock *Load Diffusion Model* node).
- **Gradio** — public `https://….gradio.live` URL, two tabs, both through the ComfyUI API:
  - **Text to Image** — prompt, size (up to 2048² native 2K), steps / cfg / seed, **number of images**
    (same prompt with seeds seed, seed+1, …, queued back to back — no extra VRAM, N× the time).
  - **Image Edit** — `image_1` is the edit target (its size becomes the canvas), `image_2..4`
    are references; refer to them in the prompt as `<image1>`, `<image2>`, … Reference
    *resolution* is a pixel budget (0 = keep own size, 1024 = official default, up to 2048);
    optional custom canvas; *KV cache precision* `int8` halves the edit cache on tight VRAM.
  - *Number of images* on the edit tab too (pick the best take of the same edit). If some
    images of a batch fail or you interrupt it, the finished ones are still shown.
  - Diffusion-model dropdown listing every GGUF and safetensors file you downloaded (read live
    from ComfyUI — `.gguf` → `UnetLoaderGGUF`, `.safetensors` → stock `UNETLoader`), text-encoder
    dropdown, galleries, and a **ComfyUI host status** panel (version, GPU memory, queue, canvas
    link, interrupt). Same idea as the Wan2.2 `app_colab.py`.

Gradio can't tunnel the ComfyUI canvas itself (it needs its own websocket to localhost),
which is why the canvas stays on the Colab proxy / cloudflared and Gradio drives the API.

ComfyUI only, no Gradio:

```python
from launch_comfyui import launch
launch()                 # launch(lowvram=True) / launch(tunnel=True)
```

Logs go to `ComfyUI/comfyui.log` (`!tail -f ComfyUI/comfyui.log`), cloudflared's to
`/tmp/cloudflared.log`.

## Cell 5 (optional) – headless generation

(when the launch cell is not blocking, i.e. you used `launch()` instead of `app_gradio.py`)

```python
# text to image
!python generate.py "cinematic portrait, 85mm, film grain" --width 1024 --height 1024 --steps 25
!python generate.py "..." --count 5 --seed 42                               # seeds 42..46
!python generate.py "..." --model qwen-image-2.1-UC-BF16.gguf              # unquantized
# image edit: first --image is the target, the rest are references (<image1>, <image2> in the prompt)
!python generate.py "Keep <image1> unchanged, put the shirt from <image2> on her" \
    --image target.png --image shirt.png --resolution 1024 --cache-dtype int8
from IPython.display import Image, display
import glob; display(Image(sorted(glob.glob("outputs/*.png"))[-1]))
```

## Workflow settings (from Comfy-Org's official 2.1 template)

- **cfg 1.0**, `euler` / `simple`, **25 steps** (official pipeline: 40–50). Negative prompt is
  ignored at cfg 1 — raise cfg only if you use one.
- Native **2K**: 2048×2048 on `EmptyLatentImage`; keep multiples of 32.
- `CLIPLoader` type must be **`qwen_image`**; a `.gguf` diffusion model goes through
  **`Unet Loader (GGUF)`**, a `.safetensors` one through the stock **`Load Diffusion Model`**
  (`UNETLoader`) — `workflows/qwen_image_2.1_safetensors_{t2i,edit}.json` are the same graphs
  with that node. `generate.py` swaps node 1 by file extension, so the API JSONs stay GGUF.
- Image edit (`workflows/qwen_image_2.1_gguf_edit.json`, mirrors Comfy-Org's official edit
  template): `LoadImage` → `TextEncodeQwenImage21` (`images.image_N` + VAE), up to 16 images;
  `image_1` is the edit target and its `latent` output sets the canvas (use an
  `EmptyLatentImage` close to that size if you must force one, or the edit shifts).
  `Qwen Image 2.1 Cache` between loader and sampler sets KV-cache device/precision
  (`int8` for tight VRAM). Prompt with `<image1>`, `<image2>`, …

## Notes

- First run: ~14 GB download (5–10 min). Colab disk is fine.
- Image edit uses more VRAM than t2i (reference latents + KV cache). On T4 use `int8` cache,
  `resolution` 512–768 and one reference.
- Download fails with `404` / `EntryNotFoundError` → the model repo renamed files (it did on
  2026-09-22); `!git pull` in `/content/qwen21` for the current `download_models.py`.
- Red `TextEncodeQwenImage21` node or GGUF "unknown architecture" → an old ComfyUI / loader
  checkout; re-run `setup_colab.sh` (resets both to the pins).
- Gradio shows "ComfyUI is not responding" → the server crashed mid-job (usually RAM/VRAM OOM);
  the message includes the log tail. Restart with `app_gradio.py --restart` after lowering the load.
- `Expected weight to be of same shape as normalized_shape … [136] and normalized_shape = [128]`
  → a Q8_0 GGUF downloaded before 2026-09-21
  ([discussion #4](https://huggingface.co/abenzerps/Qwen-Image-2.1-Uncensored-GGUF/discussions/4)):
  delete it and download again.
- OOM on T4: use `launch(lowvram=True)`, stay at 1024² and `Q4_K_M`; try `--text-encoder w4a8`.
  `BF16` / `base-bf16` (14 GB) need an L4 or better, and T4 has no bf16 / fp8 support
  (possible black images); use a Q4/Q8 GGUF there.
- `fp8` gave black images before 2026-09-23 15:53 UTC
  ([discussion #22](https://huggingface.co/abenzerps/Qwen-Image-2.1-Uncensored-GGUF/discussions/22));
  the file was re-uploaded — re-run the download cell.
- Colab's proxy URL sometimes 403s after idle — re-run Cell 4 (it won't restart the
  server, just reprints the link) or use `tunnel=True`.
