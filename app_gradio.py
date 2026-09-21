#!/usr/bin/env python
"""Gradio front-end for the ComfyUI host (public *.gradio.live link, like the Wan2.2 app).

    python app_gradio.py              # starts ComfyUI if needed, then Gradio with share=True
    python app_gradio.py --lowvram    # T4
    python app_gradio.py --no-share   # local only

Gradio cannot tunnel the ComfyUI canvas itself (websocket + localhost), so this is a
prompt → image UI driving ComfyUI through its HTTP API (generate.run), with a status
panel that shows the ComfyUI host URL, GPU memory and queue.
"""
import argparse
import os
import time

import gradio as gr
import requests

from generate import run
from launch_comfyui import PORT, URLS as COMFY_URLS, launch

HOST = f"http://127.0.0.1:{PORT}"
OUT_DIR = "outputs"


# ---- ComfyUI host helpers -----------------------------------------------------------

def api(path):
    return requests.get(f"{HOST}{path}", timeout=10).json()


def model_choices(node, field):
    try:
        return api(f"/object_info/{node}")[node]["input"]["required"][field][0]
    except Exception:
        return []


def host_status():
    try:
        s = api("/system_stats")
        q = api("/queue")
    except Exception as e:
        return f"**ComfyUI host: not reachable** on `{HOST}` — {e}"
    dev = s["devices"][0]
    gb = 2**30
    lines = [
        f"**ComfyUI host:** `{HOST}`  ·  ComfyUI `{s['system'].get('comfyui_version', '?')}`  ·  "
        f"torch `{s['system'].get('pytorch_version', '?')}`",
        f"**GPU:** {dev['name']}  ·  VRAM {(dev['vram_total'] - dev['vram_free']) / gb:.1f} / "
        f"{dev['vram_total'] / gb:.1f} GB used  ·  torch alloc "
        f"{(dev['torch_vram_total'] - dev['torch_vram_free']) / gb:.1f} GB",
        f"**Queue:** {len(q.get('queue_running', []))} running · {len(q.get('queue_pending', []))} pending",
    ]
    if COMFY_URLS["proxy"]:
        lines.append(f"**Open the ComfyUI canvas (Colab proxy):** {COMFY_URLS['proxy']}")
    if COMFY_URLS["tunnel"]:
        lines.append(f"**Open the ComfyUI canvas (cloudflare):** {COMFY_URLS['tunnel']}")
    return "\n\n".join(lines)


def interrupt():
    requests.post(f"{HOST}/interrupt", timeout=10)
    return "Interrupted."


# ---- generation ---------------------------------------------------------------------

def generate(prompt, negative, width, height, steps, cfg, seed, randomize, gguf, clip,
             progress=gr.Progress()):
    if not prompt.strip():
        raise gr.Error("Prompt is empty.")
    seed = None if randomize else int(seed)
    progress(0, desc="queued")
    t0 = time.time()
    files, used_seed = run(prompt, negative, int(width), int(height), int(steps), float(cfg),
                           seed, gguf=gguf or None, clip=clip or None, host=HOST, out=OUT_DIR)
    info = f"seed **{used_seed}** · {int(width)}×{int(height)} · {int(steps)} steps · cfg {cfg} · {time.time() - t0:.0f}s"
    return files, used_seed, info


# ---- UI -----------------------------------------------------------------------------

def build_ui():
    ggufs = model_choices("UnetLoaderGGUF", "unet_name")
    clips = model_choices("CLIPLoader", "clip_name")
    default_gguf = next((g for g in ggufs if "Q4_K_M" in g), ggufs[0] if ggufs else None)
    default_clip = next((c for c in clips if "qwen3vl" in c and "int8" in c), clips[0] if clips else None)

    with gr.Blocks(title="Qwen-Image 2.1 Uncensored (GGUF) · ComfyUI") as demo:
        gr.Markdown("# Qwen-Image 2.1 Uncensored (GGUF) — ComfyUI host")

        with gr.Accordion("ComfyUI host status", open=True):
            status = gr.Markdown(host_status())
            with gr.Row():
                gr.Button("Refresh status", size="sm").click(host_status, outputs=status)
                gr.Button("Interrupt current job", size="sm", variant="stop").click(interrupt, outputs=status)

        with gr.Row():
            with gr.Column(scale=3):
                prompt = gr.Textbox(label="Prompt", lines=5,
                                    value="Cinematic photo of a woman in a red dress on a rooftop at dusk, "
                                          "city lights bokeh, 85mm lens, shallow depth of field, film grain.")
                negative = gr.Textbox(label="Negative prompt (ignored while cfg = 1)", lines=2)
                with gr.Row():
                    width = gr.Slider(512, 2048, 1024, step=32, label="Width")
                    height = gr.Slider(512, 2048, 1024, step=32, label="Height")
                with gr.Row():
                    steps = gr.Slider(4, 60, 25, step=1, label="Steps (official: 40–50)")
                    cfg = gr.Slider(1.0, 7.0, 1.0, step=0.1, label="CFG (1.0 = official path)")
                with gr.Row():
                    seed = gr.Number(value=0, precision=0, label="Seed")
                    randomize = gr.Checkbox(True, label="Randomize seed")
                with gr.Row():
                    gguf = gr.Dropdown(ggufs, value=default_gguf, label="Diffusion model (GGUF)")
                    clip = gr.Dropdown(clips, value=default_clip, label="Text encoder")
                btn = gr.Button("Generate", variant="primary")
            with gr.Column(scale=4):
                gallery = gr.Gallery(label="Output", columns=1, height=640, object_fit="contain")
                info = gr.Markdown()

        btn.click(generate, [prompt, negative, width, height, steps, cfg, seed, randomize, gguf, clip],
                  [gallery, seed, info])
        gr.Markdown("Presets: **1K** 1024×1024 · **2K native** 2048×2048 · portrait 1024×1536 · "
                    "landscape 1536×1024 (multiples of 32). Outputs also land in `ComfyUI/output/`.")
    return demo


def main(comfy_dir="ComfyUI", lowvram=False, tunnel=False, share=True, port=7860, restart=False):
    launch(comfy_dir, lowvram, tunnel, restart=restart)
    demo = build_ui()
    demo.queue().launch(share=share, server_name="127.0.0.1", server_port=port,
                        allowed_paths=[os.path.abspath(OUT_DIR)])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--comfy-dir", default="ComfyUI")
    ap.add_argument("--lowvram", action="store_true")
    ap.add_argument("--tunnel", action="store_true", help="also open a cloudflared tunnel to ComfyUI")
    ap.add_argument("--no-share", action="store_true", help="don't create a *.gradio.live link")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--restart", action="store_true", help="restart a running ComfyUI first")
    a = ap.parse_args()
    main(a.comfy_dir, a.lowvram, a.tunnel, not a.no_share, a.port, a.restart)
