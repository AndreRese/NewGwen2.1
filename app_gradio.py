#!/usr/bin/env python
"""Gradio front-end for the ComfyUI host (public *.gradio.live link, like the Wan2.2 app).

    python app_gradio.py              # starts ComfyUI if needed, then Gradio with share=True
    python app_gradio.py --lowvram    # T4
    python app_gradio.py --restart    # reload a running ComfyUI first
    python app_gradio.py --no-share   # local only

Gradio cannot tunnel the ComfyUI canvas itself (websocket + localhost), so this is a
prompt → image UI driving ComfyUI through its HTTP API (generate.run / generate.edit),
with a status panel that shows the ComfyUI host URL, GPU memory and queue.
Tabs: Text to Image · Image Edit (image_1 = target, up to 4 refs, <imageN> in the prompt).
The diffusion-model dropdown lists both GGUF quants (UnetLoaderGGUF) and the unquantized
safetensors originals (UNETLoader); "Number of images" queues N seeds back to back.
"""
import argparse
import os
import time

import gradio as gr
import requests

from generate import ComfyError, edit, run
from launch_comfyui import PORT, URLS as COMFY_URLS, launch

HOST = f"http://127.0.0.1:{PORT}"
OUT_DIR = "outputs"
MAX_COUNT = 10


# ---- ComfyUI host helpers -----------------------------------------------------------

def api(path):
    return requests.get(f"{HOST}{path}", timeout=10).json()


def model_choices(node, field):
    try:
        return api(f"/object_info/{node}")[node]["input"]["required"][field][0]
    except Exception:
        return []


def diffusion_choices():
    """GGUF quants (leejet loader) + safetensors originals (stock UNETLoader), deduplicated."""
    seen, out = set(), []
    for name in model_choices("UnetLoaderGGUF", "unet_name") + model_choices("UNETLoader", "unet_name"):
        if name not in seen and name.endswith((".gguf", ".safetensors")):
            seen.add(name)
            out.append(name)
    return out


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
    """Drop everything still pending (a multi-image batch), then stop the running job."""
    requests.post(f"{HOST}/queue", json={"clear": True}, timeout=10)
    requests.post(f"{HOST}/interrupt", timeout=10)
    return "Queue cleared + current job interrupted.\n\n" + host_status()


# ---- generation ---------------------------------------------------------------------

def _info(seed, count, extra, t0):
    seeds = f"seed **{seed}**" if count == 1 else f"seeds **{seed}–{(seed + count - 1) % 2**32}**"
    return f"{seeds} · {extra} · {time.time() - t0:.0f}s"


def _progress_cb(progress, count):
    def cb(done, total):
        progress(done / total, desc=f"{done}/{total} done")
    return cb if count > 1 else None


def generate_t2i(prompt, negative, width, height, steps, cfg, seed, randomize, count, model, clip,
                 progress=gr.Progress()):
    if not prompt.strip():
        raise gr.Error("Prompt is empty.")
    count = int(count)
    progress(0, desc=f"queued ({count})" if count > 1 else "queued")
    t0 = time.time()
    try:
        files, used_seed = run(prompt, negative, int(width), int(height), int(steps), float(cfg),
                               None if randomize else int(seed), model=model or None, clip=clip or None,
                               host=HOST, out=OUT_DIR, count=count, progress=_progress_cb(progress, count))
    except ComfyError as e:
        raise gr.Error(str(e))
    return files, used_seed, _info(used_seed, count, f"{len(files)} image(s) · {int(width)}×{int(height)} · "
                                                     f"{int(steps)} steps · cfg {cfg} · {model}", t0)


def generate_edit(prompt, negative, img1, img2, img3, img4, resolution, custom_size, width, height,
                  steps, cfg, seed, randomize, count, cache_dtype, model, clip, progress=gr.Progress()):
    if not prompt.strip():
        raise gr.Error("Prompt is empty.")
    if not img1:
        raise gr.Error("image_1 (the edit target) is required.")
    images = [p for p in (img1, img2, img3, img4) if p]
    count = int(count)
    progress(0, desc="uploading + queued")
    t0 = time.time()
    try:
        files, used_seed = edit(prompt, images, negative, int(resolution),
                                int(width) if custom_size else None, int(height) if custom_size else None,
                                int(steps), float(cfg), None if randomize else int(seed),
                                model=model or None, clip=clip or None, cache_dtype=cache_dtype,
                                host=HOST, out=OUT_DIR, count=count, progress=_progress_cb(progress, count))
    except ComfyError as e:
        raise gr.Error(str(e))
    canvas = f"{int(width)}×{int(height)}" if custom_size else "image_1 size"
    return files, used_seed, _info(used_seed, count, f"{len(files)} image(s) · {len(images)} input(s) · "
                                                     f"res {int(resolution)} · canvas {canvas} · "
                                                     f"{int(steps)} steps · cfg {cfg} · {model}", t0)


# ---- UI -----------------------------------------------------------------------------

def sampler_controls(default_steps=25):
    with gr.Row():
        steps = gr.Slider(4, 60, default_steps, step=1, label="Steps (official: 40–50)")
        cfg = gr.Slider(1.0, 7.0, 1.0, step=0.1, label="CFG (1.0 = official path)")
    with gr.Row():
        seed = gr.Number(value=0, precision=0, label="Seed")
        randomize = gr.Checkbox(True, label="Randomize seed")
        count = gr.Slider(1, MAX_COUNT, 1, step=1, label="Number of images (seed, seed+1, …)")
    return steps, cfg, seed, randomize, count


def build_ui():
    models = diffusion_choices()
    clips = model_choices("CLIPLoader", "clip_name")
    default_model = next((m for m in models if "Q4_K_M" in m), models[0] if models else None)
    default_clip = next((c for c in clips if "qwen3vl" in c and "int8" in c), clips[0] if clips else None)

    with gr.Blocks(title="Qwen-Image 2.1 (GGUF / original) · ComfyUI") as demo:
        gr.Markdown("# Qwen-Image 2.1 — ComfyUI host")

        with gr.Accordion("ComfyUI host status", open=True):
            status = gr.Markdown(host_status())
            with gr.Row():
                gr.Button("Refresh status", size="sm").click(host_status, outputs=status)
                gr.Button("Interrupt (clears the queue)", size="sm", variant="stop").click(interrupt, outputs=status)

        with gr.Row():
            model = gr.Dropdown(models, value=default_model,
                                label="Diffusion model (*.gguf = GGUF quant · *.safetensors = original weights)")
            clip = gr.Dropdown(clips, value=default_clip, label="Text encoder")

        with gr.Tabs():
            # ---------------------------------------------------------------- text to image
            with gr.Tab("Text to Image"):
                with gr.Row():
                    with gr.Column(scale=3):
                        prompt = gr.Textbox(label="Prompt", lines=5,
                                            value="Cinematic photo of a woman in a red dress on a rooftop at dusk, "
                                                  "city lights bokeh, 85mm lens, shallow depth of field, film grain.")
                        negative = gr.Textbox(label="Negative prompt (ignored while cfg = 1)", lines=2)
                        with gr.Row():
                            width = gr.Slider(512, 2048, 1024, step=32, label="Width")
                            height = gr.Slider(512, 2048, 1024, step=32, label="Height")
                        steps, cfg, seed, randomize, count = sampler_controls()
                        btn = gr.Button("Generate", variant="primary")
                    with gr.Column(scale=4):
                        gallery = gr.Gallery(label="Output", columns=2, height=640, object_fit="contain")
                        info = gr.Markdown()
                btn.click(generate_t2i,
                          [prompt, negative, width, height, steps, cfg, seed, randomize, count, model, clip],
                          [gallery, seed, info])
                gr.Markdown("Presets: **1K** 1024×1024 · **2K native** 2048×2048 · portrait 1024×1536 · "
                            "landscape 1536×1024 (multiples of 32). *Number of images* runs the same prompt "
                            "with consecutive seeds, one after another (no extra VRAM); the seed box shows the "
                            "first one. Outputs also land in `ComfyUI/output/`.")

            # ---------------------------------------------------------------- image edit
            with gr.Tab("Image Edit"):
                with gr.Row():
                    with gr.Column(scale=3):
                        e_prompt = gr.Textbox(
                            label="Edit instruction — refer to images as <image1>, <image2>, …", lines=4,
                            value="Keep the character and pose in <image1> unchanged, change the background to a sunny beach.")
                        e_negative = gr.Textbox(label="Negative prompt (ignored while cfg = 1)", lines=2)
                        with gr.Row():
                            img1 = gr.Image(type="filepath", label="image_1 — edit target (sets the canvas)")
                            img2 = gr.Image(type="filepath", label="image_2 — reference (optional)")
                        with gr.Row():
                            img3 = gr.Image(type="filepath", label="image_3 — reference (optional)")
                            img4 = gr.Image(type="filepath", label="image_4 — reference (optional)")
                        resolution = gr.Dropdown([0, 512, 768, 1024, 1536, 2048], value=0,
                                                 label="Reference resolution (pixel budget; 0 = keep own size, 1024 = official)")
                        with gr.Row():
                            custom_size = gr.Checkbox(False, label="Custom canvas (else follows image_1)")
                            e_width = gr.Slider(512, 2048, 1024, step=32, label="Width")
                            e_height = gr.Slider(512, 2048, 1024, step=32, label="Height")
                        e_steps, e_cfg, e_seed, e_randomize, e_count = sampler_controls()
                        cache_dtype = gr.Radio(["default", "int8", "int4"], value="default",
                                               label="KV cache precision (int8 halves it — use on T4/15 GB)")
                        e_btn = gr.Button("Edit", variant="primary")
                    with gr.Column(scale=4):
                        e_gallery = gr.Gallery(label="Output", columns=2, height=640, object_fit="contain")
                        e_info = gr.Markdown()
                e_btn.click(generate_edit,
                            [e_prompt, e_negative, img1, img2, img3, img4, resolution, custom_size, e_width, e_height,
                             e_steps, e_cfg, e_seed, e_randomize, e_count, cache_dtype, model, clip],
                            [e_gallery, e_seed, e_info])
                gr.Markdown("image_1 is what gets edited and sets the output size; the others are references "
                            "(clothes, style, a second character…). Keep a custom canvas close to image_1's size "
                            "or the edit shifts. *Number of images* = the same edit with consecutive seeds, so you "
                            "can pick the best take. The ComfyUI canvas workflow `qwen_image_2.1_gguf_edit` takes up to 16 images.")
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
