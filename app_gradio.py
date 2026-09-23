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
The diffusion-model dropdown lists every GGUF (UnetLoaderGGUF) and safetensors (UNETLoader)
file; "Number of images" queues N seeds back to back and keeps the finished ones if some fail.
Click a result to select it: "Send to Edit" makes it image_1, "Use this seed" reproduces it.
The edit tab shows the output size image_1 will get and warns before it is downscaled.
Up to 3 LoRAs (Qwen-Image 2.1 only) apply to both tabs; they can be downloaded by URL.
"""
import argparse
import os
import time

import gradio as gr
import requests
from PIL import Image

from download_models import LoraError, download_lora
from generate import ComfyError, edit, edit_canvas, run, seed_of
from launch_comfyui import PORT, URLS as COMFY_URLS, launch

HOST = f"http://127.0.0.1:{PORT}"
OUT_DIR = "outputs"
COMFY_DIR = "ComfyUI"  # set by main(); LoRA downloads go to COMFY_DIR/models/loras
MAX_COUNT = 10
N_LORAS = 3
NO_LORA = "(none)"
NATIVE_PIXELS = 2048 * 2048


# ---- ComfyUI host helpers -----------------------------------------------------------

def api(path):
    return requests.get(f"{HOST}{path}", timeout=10).json()


def model_choices(node, field):
    try:
        return api(f"/object_info/{node}")[node]["input"]["required"][field][0]
    except Exception:
        return []


def diffusion_choices():
    """GGUF quants (leejet loader) + safetensors (stock UNETLoader), deduplicated."""
    seen, out = set(), []
    for name in model_choices("UnetLoaderGGUF", "unet_name") + model_choices("UNETLoader", "unet_name"):
        if name not in seen and name.endswith((".gguf", ".safetensors")):
            seen.add(name)
            out.append(name)
    return out


def lora_choices():
    return [NO_LORA] + model_choices("LoraLoaderModelOnly", "lora_name")


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
    try:
        requests.post(f"{HOST}/queue", json={"clear": True}, timeout=10)
        requests.post(f"{HOST}/interrupt", timeout=10)
    except requests.RequestException as e:
        return f"Could not reach ComfyUI: {e}\n\n" + host_status()
    return "Queue cleared + current job interrupted.\n\n" + host_status()


def _keep(value, choices, fallback):
    return value if value in choices else fallback


def refresh_lists(model, clip, lora1, lora2, lora3):
    """Re-read the model / text-encoder / LoRA file lists from ComfyUI (after a download)."""
    models, clips, lchoices = diffusion_choices(), model_choices("CLIPLoader", "clip_name"), lora_choices()
    return ([gr.update(choices=models, value=_keep(model, models, models[0] if models else None)),
             gr.update(choices=clips, value=_keep(clip, clips, clips[0] if clips else None))]
            + [gr.update(choices=lchoices, value=_keep(v, lchoices, NO_LORA)) for v in (lora1, lora2, lora3)])


# ---- LoRAs --------------------------------------------------------------------------

def _loras(args):
    """(name1, strength1, name2, strength2, ...) from the UI -> [(name, strength)]."""
    pairs = zip(args[0::2], args[1::2])
    return [(n, float(s)) for n, s in pairs if n and n != NO_LORA and float(s) != 0.0]


def _lora_text(loras):
    return " · LoRA " + ", ".join(f"{n.rsplit('.', 1)[0]}×{s:g}" for n, s in loras) if loras else ""


def lora_download(url, token, lora1, lora2, lora3, progress=gr.Progress()):
    """Download a LoRA by URL, refresh the dropdowns and put it in the first free slot."""
    if not (url or "").strip():
        raise gr.Error("Paste a LoRA link first.")

    def cb(done, total):
        progress(done / total if total else None, desc=f"{done / 1e6:.0f} MB")

    try:
        name = download_lora(url, COMFY_DIR, token=(token or "").strip() or None, progress=cb)
    except (LoraError, requests.RequestException) as e:
        raise gr.Error(f"LoRA download failed: {e}")
    lchoices = lora_choices()
    if name not in lchoices:  # ComfyUI caches file lists briefly; show it anyway
        lchoices.append(name)
    values = [lora1, lora2, lora3]
    if name not in values:
        free = next((i for i, v in enumerate(values) if v in (None, NO_LORA)), None)
        if free is not None:
            values[free] = name
    msg = f"Downloaded **{name}**" + ("" if name in values else " — pick it in a LoRA slot")
    return [msg] + [gr.update(choices=lchoices, value=v) for v in values]


# ---- results: gallery, selection, send to edit ----------------------------------------

def _gallery(files):
    return [(f, f"seed {seed_of(f)}" if seed_of(f) is not None else os.path.basename(f)) for f in files]


def _sel_text(results, i):
    if not results:
        return ""
    f = results[i]
    return f"Selected **#{i + 1}** · seed **{seed_of(f)}** — click another image to change"


def on_select(results, evt: gr.SelectData):
    return evt.index, _sel_text(results, evt.index)


def _selected(results, selected):
    if not results:
        raise gr.Error("Generate an image first.")
    return results[min(int(selected or 0), len(results) - 1)]


def send_to_edit(results, selected):
    return _selected(results, selected), gr.Tabs(selected="edit")


def use_as_image1(results, selected):
    return _selected(results, selected)


def use_seed(results, selected):
    s = seed_of(_selected(results, selected))
    if s is None:
        raise gr.Error("This file name carries no seed.")
    return s, False


# ---- generation ---------------------------------------------------------------------

def _info(seed, count, extra, t0):
    seeds = f"seed **{seed}**" if count == 1 else f"seeds **{seed}–{(seed + count - 1) % 2**32}**"
    return f"{seeds} · {extra} · {time.time() - t0:.0f}s"


def _done(files, seed, info):
    """Outputs of both generate functions: gallery, seed box, info, results, selected index, selection text."""
    return _gallery(files), seed, info, files, 0, _sel_text(files, 0)


def _partial(e, count, extra, t0):
    """A batch where some images failed: show the finished ones with a warning, or raise
    if nothing finished."""
    if not e.files:
        raise gr.Error(str(e))
    gr.Warning(str(e).splitlines()[0] + " — see the note under the gallery.")
    detail = "\n".join(f"- {line}" for line in str(e).splitlines()[1:])
    return _done(e.files, e.seed,
                 _info(e.seed, count, extra, t0) + f"\n\n**⚠ {len(e.files)} of {count} finished:**\n{detail}")


def _progress_cb(progress, count):
    def cb(done, total):
        progress(done / total, desc=f"{done}/{total} done")
    return cb if count > 1 else None


def generate_t2i(prompt, negative, width, height, steps, cfg, seed, randomize, count, model, clip,
                 l1, s1, l2, s2, l3, s3, progress=gr.Progress()):
    if not prompt.strip():
        raise gr.Error("Prompt is empty.")
    count = int(count)
    loras = _loras([l1, s1, l2, s2, l3, s3])
    progress(0, desc=f"queued ({count})" if count > 1 else "queued")
    t0 = time.time()
    extra = f"{int(width)}×{int(height)} · {int(steps)} steps · cfg {cfg} · {model}{_lora_text(loras)}"
    try:
        files, used_seed = run(prompt, negative, int(width), int(height), int(steps), float(cfg),
                               None if randomize else int(seed), model=model or None, clip=clip or None,
                               host=HOST, out=OUT_DIR, count=count, progress=_progress_cb(progress, count),
                               loras=loras)
    except ComfyError as e:
        return _partial(e, count, extra, t0)
    return _done(files, used_seed, _info(used_seed, count, f"{len(files)} image(s) · {extra}", t0))


def generate_edit(prompt, negative, img1, img2, img3, img4, resolution, custom_size, width, height,
                  steps, cfg, seed, randomize, count, cache_dtype, model, clip,
                  l1, s1, l2, s2, l3, s3, progress=gr.Progress()):
    if not prompt.strip():
        raise gr.Error("Prompt is empty.")
    if not img1:
        raise gr.Error("image_1 (the edit target) is required.")
    images = [p for p in (img1, img2, img3, img4) if p]
    count = int(count)
    loras = _loras([l1, s1, l2, s2, l3, s3])
    progress(0, desc="uploading + queued")
    t0 = time.time()
    canvas = f"{int(width)}×{int(height)}" if custom_size else "image_1 size"
    extra = (f"{len(images)} input(s) · res {int(resolution)} · canvas {canvas} · "
             f"{int(steps)} steps · cfg {cfg} · {model}{_lora_text(loras)}")
    try:
        files, used_seed = edit(prompt, images, negative, int(resolution),
                                int(width) if custom_size else None, int(height) if custom_size else None,
                                int(steps), float(cfg), None if randomize else int(seed),
                                model=model or None, clip=clip or None, cache_dtype=cache_dtype,
                                host=HOST, out=OUT_DIR, count=count, progress=_progress_cb(progress, count),
                                loras=loras)
    except ComfyError as e:
        return _partial(e, count, extra, t0)
    return _done(files, used_seed, _info(used_seed, count, f"{len(files)} image(s) · {extra}", t0))


# ---- edit-tab helpers ---------------------------------------------------------------

def _image_size(path):
    try:
        with Image.open(path) as im:
            return im.size
    except Exception:
        return None


def canvas_info(img1, resolution, custom_size, width, height):
    """What TextEncodeQwenImage21 will do with image_1, and warnings before quality is lost."""
    if not img1:
        return "*Upload image_1 to see the output size.*"
    size = _image_size(img1)
    if not size:
        return "⚠ Could not read image_1."
    w, h = size
    res = int(resolution)
    cw, ch = edit_canvas(w, h, res)
    scale = ((cw * ch) / (w * h)) ** 0.5
    lines = [f"**image_1** {w}×{h} → **{cw}×{ch}** at resolution {res}"
             + (" (own size)" if res == 0 else "")
             + (f" · output canvas **{int(width)}×{int(height)}** (custom)" if custom_size
                else " · this is the output size")]
    if scale < 0.9:
        lines.append(f"⚠ image_1 is **downscaled ×{scale:.2f}** — the result comes out smaller and softer "
                     "than your source. Use resolution **0** (keep) or a higher one.")
    elif scale > 1.1:
        lines.append(f"ℹ image_1 is upscaled ×{scale:.2f} to reach the budget — it can't gain detail it doesn't have.")
    if cw * ch > NATIVE_PIXELS * 1.05:
        lines.append(f"⚠ {cw * ch / 1e6:.1f} MP is above native 2K (4.2 MP): slow and may run out of VRAM — "
                     "pick resolution 2048.")
    if custom_size:
        ratio, cratio = cw / ch, int(width) / int(height)
        if abs(cratio - ratio) / ratio > 0.02:
            lines.append("⚠ Custom canvas has a different aspect ratio than image_1 — the edit will shift or "
                         "stretch. Untick and re-tick *Custom canvas* to start from image_1's size.")
    return "\n\n".join(lines)


def fill_canvas(img1, resolution, custom_size, width, height):
    """Ticking 'Custom canvas' starts the sliders at image_1's canvas size."""
    size = _image_size(img1) if (custom_size and img1) else None
    if not size:
        return width, height
    cw, ch = edit_canvas(*size, int(resolution))
    return min(max(cw, 512), 2048), min(max(ch, 512), 2048)


def preset_fast():
    return 25, 1024, "default"


def preset_quality(img1):
    """40 steps and image_1 at full size, capped at native 2K."""
    res = gr.update()
    size = _image_size(img1) if img1 else None
    if size:
        cw, ch = edit_canvas(*size, 0)
        res = 0 if cw * ch <= NATIVE_PIXELS * 1.05 else 2048
    return 40, res, "default"


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


def results_panel():
    """Gallery + selection state + action row under it."""
    gallery = gr.Gallery(label="Output — click an image to select it", columns=2, height=640, object_fit="contain")
    info = gr.Markdown()
    results = gr.State([])
    selected = gr.State(0)
    sel_md = gr.Markdown()
    gallery.select(on_select, [results], [selected, sel_md])
    return gallery, info, results, selected, sel_md


def build_ui():
    models = diffusion_choices()
    clips = model_choices("CLIPLoader", "clip_name")
    lchoices = lora_choices()
    default_model = next((m for m in models if "UC-Q4_K_M" in m),
                         next((m for m in models if "Q4_K_M" in m), models[0] if models else None))
    default_clip = next((c for c in clips if "qwen3vl" in c and "int8" in c), clips[0] if clips else None)

    with gr.Blocks(title="Qwen-Image 2.1 (GGUF / original) · ComfyUI") as demo:
        gr.Markdown("# Qwen-Image 2.1 — ComfyUI host")

        with gr.Accordion("ComfyUI host status", open=True):
            status = gr.Markdown(host_status())
            with gr.Row():
                gr.Button("Refresh status", size="sm").click(host_status, outputs=status)
                gr.Button("Interrupt (clears the queue)", size="sm", variant="stop").click(interrupt, outputs=status)

        with gr.Row():
            model = gr.Dropdown(models, value=default_model, scale=4,
                                label="Diffusion model (UC = uncensored · *.gguf → GGUF loader · *.safetensors → UNETLoader)")
            clip = gr.Dropdown(clips, value=default_clip, label="Text encoder", scale=3)
            refresh = gr.Button("↻ Refresh lists", size="sm", scale=1)

        with gr.Accordion("LoRAs — Qwen-Image 2.1 LoRAs only, applied to both tabs", open=False):
            lora_slots = []
            for i in range(1, N_LORAS + 1):
                with gr.Row():
                    name = gr.Dropdown(lchoices, value=NO_LORA, label=f"LoRA {i}", scale=3)
                    strength = gr.Slider(-2.0, 2.0, 1.0, step=0.05, label="strength", scale=2)
                lora_slots += [name, strength]
            lora_names = lora_slots[0::2]
            with gr.Row():
                lora_url = gr.Textbox(label="Download a LoRA by URL", scale=4,
                                      placeholder="https://huggingface.co/<user>/<repo>/blob/main/x.safetensors  or  "
                                                  "https://civitai.com/api/download/models/<id>")
                lora_token = gr.Textbox(label="Token (optional: Civitai API key / HF token)", type="password", scale=2)
                lora_btn = gr.Button("Download", scale=1)
            lora_status = gr.Markdown("Files go to `ComfyUI/models/loras` (only `.safetensors`). A LoRA made for "
                                      "Qwen-Image 1.x / 2.0 loads without error but does nothing — check the "
                                      "ComfyUI log for `lora key not loaded`.")
            lora_btn.click(lora_download, [lora_url, lora_token] + lora_names, [lora_status] + lora_names)
        refresh.click(refresh_lists, [model, clip] + lora_names, [model, clip] + lora_names)

        with gr.Tabs() as tabs:
            # ---------------------------------------------------------------- text to image
            with gr.Tab("Text to Image", id="t2i"):
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
                        gallery, info, results, selected, sel_md = results_panel()
                        with gr.Row():
                            to_edit = gr.Button("✎ Send to Edit (as image_1)", size="sm")
                            t_seed = gr.Button("Use this seed", size="sm")
                gr.Markdown("Presets: **1K** 1024×1024 · **2K native** 2048×2048 · portrait 1024×1536 · "
                            "landscape 1536×1024 (multiples of 32). *Number of images* runs the same prompt "
                            "with consecutive seeds, one after another (no extra VRAM); each image shows its "
                            "seed. Outputs also land in `ComfyUI/output/`.")

            # ---------------------------------------------------------------- image edit
            with gr.Tab("Image Edit", id="edit"):
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
                        with gr.Row():
                            fast = gr.Button("Fast preset (25 steps · res 1024)", size="sm")
                            quality = gr.Button("Quality preset (40 steps · image_1 full size ≤ 2K)", size="sm")
                        resolution = gr.Dropdown([0, 512, 768, 1024, 1536, 2048], value=0,
                                                 label="Reference resolution (pixel budget; 0 = keep own size, 1024 = official)")
                        with gr.Row():
                            custom_size = gr.Checkbox(False, label="Custom canvas (else follows image_1)")
                            e_width = gr.Slider(512, 2048, 1024, step=32, label="Width")
                            e_height = gr.Slider(512, 2048, 1024, step=32, label="Height")
                        canvas_md = gr.Markdown(canvas_info(None, 0, False, 1024, 1024))
                        e_steps, e_cfg, e_seed, e_randomize, e_count = sampler_controls()
                        cache_dtype = gr.Radio(["default", "int8", "int4"], value="default",
                                               label="KV cache precision (int8 halves it — use on T4/15 GB)")
                        e_btn = gr.Button("Edit", variant="primary")
                    with gr.Column(scale=4):
                        e_gallery, e_info, e_results, e_selected, e_sel_md = results_panel()
                        with gr.Row():
                            again = gr.Button("↺ Use as image_1 (edit it again)", size="sm")
                            e_seed_btn = gr.Button("Use this seed", size="sm")
                gr.Markdown("image_1 is what gets edited and sets the output size; the others are references "
                            "(clothes, style, a second character…). Keep a custom canvas close to image_1's size "
                            "or the edit shifts. *Number of images* = the same edit with consecutive seeds, so you "
                            "can pick the best take. Chained edits lose a little detail each round (VAE round trip) — "
                            "go back to the original when you can. The ComfyUI canvas workflow "
                            "`qwen_image_2.1_gguf_edit` takes up to 16 images.")

        # ---- wiring
        btn.click(generate_t2i,
                  [prompt, negative, width, height, steps, cfg, seed, randomize, count, model, clip] + lora_slots,
                  [gallery, seed, info, results, selected, sel_md])
        e_btn.click(generate_edit,
                    [e_prompt, e_negative, img1, img2, img3, img4, resolution, custom_size, e_width, e_height,
                     e_steps, e_cfg, e_seed, e_randomize, e_count, cache_dtype, model, clip] + lora_slots,
                    [e_gallery, e_seed, e_info, e_results, e_selected, e_sel_md])
        to_edit.click(send_to_edit, [results, selected], [img1, tabs])
        t_seed.click(use_seed, [results, selected], [seed, randomize])
        again.click(use_as_image1, [e_results, e_selected], [img1])
        e_seed_btn.click(use_seed, [e_results, e_selected], [e_seed, e_randomize])

        canvas_inputs = [img1, resolution, custom_size, e_width, e_height]
        for comp in (img1, resolution, e_width, e_height):
            comp.change(canvas_info, canvas_inputs, canvas_md)
        custom_size.change(fill_canvas, canvas_inputs, [e_width, e_height]).then(canvas_info, canvas_inputs, canvas_md)
        fast.click(preset_fast, None, [e_steps, resolution, cache_dtype])
        quality.click(preset_quality, [img1], [e_steps, resolution, cache_dtype])
    return demo


def main(comfy_dir="ComfyUI", lowvram=False, tunnel=False, share=True, port=7860, restart=False):
    global COMFY_DIR
    COMFY_DIR = comfy_dir
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
