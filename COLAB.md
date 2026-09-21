# Running on Google Colab (ComfyUI)

Runs **Qwen-Image 2.1 Uncensored (GGUF)** inside ComfyUI on a Colab GPU. Same idea as the
Wan2.2 project's `COLAB.md`, but ComfyUI is the app: no Gradio port, you get the ComfyUI
web UI through Colab's port proxy (or a cloudflared tunnel), plus `generate.py` for
headless runs from a cell.

The whole thing is also packaged as a notebook: `Qwen_Image_2.1_GGUF_ComfyUI.ipynb`.

**Runtime → Change runtime type → GPU.** Check with `!nvidia-smi`.

| GPU | VRAM / RAM | Recommended | Notes |
|---|---|---|---|
| T4 (free) | 15 GB / 12.7 GB | `Q4_K_M` + `int8` encoder, `--lowvram` | Works but tight — the 9 GB text encoder is mmap'd from disk and offloaded after encoding. Slow (no bf16 on sm75). |
| L4 | 22.5 GB / 53 GB | `Q4_K_M`…`Q8_0` + `int8` encoder | Sweet spot. Everything stays resident. |
| A100 40/80 GB | 40–80 GB / 83 GB | `Q8_0` + `bf16` encoder | Fastest; at this point you could also use Comfy-Org's `bf16` safetensors instead of GGUF. |

## Cell 1 – get the code

```python
!git clone https://github.com/AndreRese/NewGwen2.1 qwen21
%cd qwen21
```

(or upload the folder: `setup_colab.sh`, `download_models.py`, `launch_comfyui.py`,
`generate.py`, `colab_requirements.txt`, `packages.txt` and `workflows/`)

## Cell 2 – ComfyUI + custom nodes + dependencies

```python
!bash setup_colab.sh
```

Clones ComfyUI **master** (the `TextEncodeQwenImage21` node is not in a tagged release yet)
and the **leejet fork** of ComfyUI-GGUF — as of 2026-09-20 it is the only GGUF loader
with Qwen-Image 2.1 support (city96 upstream hasn't merged it). Installs both
`requirements.txt` files plus `colab_requirements.txt`, and copies the workflow into
`ComfyUI/user/default/workflows/` so it shows up in the UI's workflow browser.

## Cell 3 – models (~14 GB for the default combo)

```python
!python download_models.py --quant Q4_K_M --text-encoder int8
```

No HF token needed — all three files come from `abenzerps/Qwen-Image-2.1-Uncensored-GGUF`,
which mirrors the Comfy-Org text encoder and VAE. Options:

| flag | choices | size |
|---|---|---|
| `--quant` | `Q4_0` 4.05 GB · **`Q4_K_M` 4.60 GB** · `Q8_0` 7.59 GB (only one is downloaded) | diffusion model |
| `--text-encoder` | **`int8`** 9.35 GB · `bf16` 17.5 GB · `w4a8` 6.3 GB (from Comfy-Org, experimental) | Qwen3-VL 8B |

Files land in:

```
ComfyUI/models/
├── diffusion_models/qwen-image-2.1-Q4_K_M.gguf
├── text_encoders/qwen3vl_8b_int8_convrot.safetensors
└── vae/qwen_image_2.1_vae_bf16.safetensors
```

## Cell 4 – launch ComfyUI

```python
from launch_comfyui import launch
launch()                 # L4 / A100
# launch(lowvram=True)   # T4
# launch(tunnel=True)    # also print a *.trycloudflare.com URL
```

Prints a `https://….colab.googleusercontent.com` link (Colab's port proxy — only works
while you're logged into that Colab session). Open it, then **Workflow → Open** and pick
`qwen_image_2.1_gguf_t2i`, or drag `workflows/qwen_image_2.1_gguf_t2i.json` onto the canvas.

Logs go to `ComfyUI/comfyui.log` (`!tail -f ComfyUI/comfyui.log`).

## Cell 5 (optional) – headless generation

```python
!python generate.py "cinematic portrait, 85mm, film grain" --width 1024 --height 1024 --steps 25
from IPython.display import Image, display
import glob; display(Image(sorted(glob.glob("outputs/*.png"))[-1]))
```

## Workflow settings (from Comfy-Org's official 2.1 template)

- **cfg 1.0**, `euler` / `simple`, **25 steps** (official pipeline: 40–50). Negative prompt is
  ignored at cfg 1 — raise cfg only if you use one.
- Native **2K**: 2048×2048 on `EmptyLatentImage`; keep multiples of 32.
- `CLIPLoader` type must be **`qwen_image`**; the diffusion model goes through
  **`Unet Loader (GGUF)`**.
- Image edit: the same `TextEncodeQwenImage21` node takes up to 16 reference images
  (`images.image_1…`) plus the VAE — wire a `LoadImage` into it and use its `latent` output
  instead of `EmptyLatentImage`.

## Notes

- First run: ~14 GB download (5–10 min with `hf_transfer`). Colab disk is fine.
- If the GGUF loader errors with an unknown architecture, `setup_colab.sh` pulled an old
  checkout — re-run it, or `!git -C ComfyUI/custom_nodes/ComfyUI-GGUF pull`.
- If `TextEncodeQwenImage21` is missing (red node), ComfyUI is too old:
  `!git -C ComfyUI pull`.
- OOM on T4: use `launch(lowvram=True)`, stay at 1024² and `Q4_K_M`; try `--text-encoder w4a8`.
- Colab's proxy URL sometimes 403s after idle — re-run Cell 4 (it won't restart the
  server, just reprints the link) or use `tunnel=True`.
