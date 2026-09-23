#!/usr/bin/env python
"""Headless generation through a running ComfyUI (no browser needed).

Text to image:
    python generate.py "a cat wearing a spacesuit, studio photo"
    python generate.py "..." --width 2048 --height 2048 --steps 40 --seed 42 --out outputs/
    python generate.py "..." --count 5                      # 5 images, seeds seed..seed+4
    python generate.py "..." --model qwen-image-2.1-UC-BF16.gguf      # uncensored, unquantized
    python generate.py "..." --model qwen-image-2.1-UC-fp8.safetensors  # safetensors -> UNETLoader

Image edit (image_1 is the edit target, the rest are references; use <image1>, <image2> in the prompt):
    python generate.py "Keep <image1> unchanged, put the shirt from <image2> on her" --image a.png --image b.png
    python generate.py "make it night" --image a.png --resolution 1024 --cache-dtype int8

Loads workflows/qwen_image_2.1_gguf_{t2i,edit}_api.json, patches the inputs, POSTs to
/prompt, waits for the job and copies the PNGs to --out. --model picks the diffusion file:
*.gguf goes through UnetLoaderGGUF, *.safetensors (fp8 / int8 / base bf16) through the
stock UNETLoader. --count N queues N jobs with consecutive seeds (same VRAM, N x time); if
some fail, the finished ones are still saved.
"""
import argparse
import copy
import hashlib
import json
import os
import random
import sys
import time
import urllib.parse

import requests

from launch_comfyui import log_tail

HERE = os.path.dirname(os.path.abspath(__file__))
T2I_JSON = os.path.join(HERE, "workflows", "qwen_image_2.1_gguf_t2i_api.json")
EDIT_JSON = os.path.join(HERE, "workflows", "qwen_image_2.1_gguf_edit_api.json")
DEFAULT_HOST = "http://127.0.0.1:8188"
COMFY_DIR = "ComfyUI"  # only used to show the server log when it crashes
HTTP_TIMEOUT = 30      # seconds per HTTP call (ComfyUI answers while it is sampling)


# ---- ComfyUI API plumbing -----------------------------------------------------------

class ComfyError(RuntimeError):
    """A job failed. `files` holds the images that did finish (partial batch), `seed` the
    first seed of the batch (set by run/edit)."""

    def __init__(self, msg, files=None, seed=None):
        super().__init__(msg)
        self.files = files or []
        self.seed = seed


class ComfyDown(ComfyError):
    """The ComfyUI server stopped answering (crashed / killed / restarting)."""


def _http(method, url, retries=3, **kw):
    """requests.request with a timeout; a server that stays unreachable becomes ComfyDown."""
    kw.setdefault("timeout", HTTP_TIMEOUT)
    for attempt in range(retries):
        try:
            return requests.request(method, url, **kw)
        except (requests.ConnectionError, requests.Timeout) as e:
            if attempt == retries - 1:
                raise ComfyDown(
                    f"ComfyUI is not responding at {url.split('/', 3)[2]} ({type(e).__name__}) — it most likely "
                    "crashed, usually out of RAM/VRAM: try LOWVRAM, a smaller model / text encoder or a lower "
                    "resolution, then restart the launch cell with RESTART = True.\n\n"
                    f"Last lines of {os.path.join(COMFY_DIR, 'comfyui.log')}:\n{log_tail(COMFY_DIR, 15)}") from e
            time.sleep(2)


def upload_image(path, host=DEFAULT_HOST):
    """Upload a local image to ComfyUI's input folder; returns the name LoadImage expects.
    The name is prefixed with a content hash: Gradio calls every pasted / webcam image
    `image.png`, and with overwrite on, two of those would otherwise become the same file."""
    with open(path, "rb") as f:
        data = f.read()
    name = f"{hashlib.sha1(data).hexdigest()[:12]}_{os.path.basename(path)}"
    r = _http("POST", f"{host}/upload/image", timeout=120, files={"image": (name, data)},
              data={"overwrite": "true", "type": "input", "subfolder": "qwen21"})
    r.raise_for_status()
    j = r.json()
    return f"{j['subfolder']}/{j['name']}" if j.get("subfolder") else j["name"]


def queue_prompt(wf, host=DEFAULT_HOST):
    """POST an API-format workflow; returns its prompt_id."""
    r = _http("POST", f"{host}/prompt", json={"prompt": wf})
    if r.status_code != 200:
        raise ComfyError(f"ComfyUI rejected the prompt ({r.status_code}):\n{r.text[:2000]}")
    pid = r.json()["prompt_id"]
    print(f">> queued {pid}")
    return pid


def _history(pid, host):
    return _http("GET", f"{host}/history/{pid}").json().get(pid)


def _in_queue(pid, host):
    q = _http("GET", f"{host}/queue").json()
    return any(item[1] == pid for item in q.get("queue_running", []) + q.get("queue_pending", []))


def _error_message(status):
    """Turn a history status block into one readable line."""
    for kind, data in status.get("messages", []):
        if kind == "execution_interrupted":
            return "interrupted"
        if kind == "execution_error":
            return f"{data.get('node_type', '?')}: {data.get('exception_type', '')} {data.get('exception_message', '')}".strip()
    return json.dumps(status, indent=2)[:2000]


def wait_prompt(pid, host=DEFAULT_HOST, out="outputs", timeout=1800):
    """Wait for a queued prompt and download its images. Returns list of paths.
    A prompt that is neither queued nor in the history was removed from the queue
    (Interrupt clears it) — that is reported at once instead of waiting for the timeout."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        hist = _history(pid, host)
        if not hist and not _in_queue(pid, host):
            hist = _history(pid, host)  # it may have finished between the two calls
            if not hist:
                raise ComfyError("cancelled (removed from the queue)")
        if hist:
            status = hist.get("status", {})
            if status.get("status_str") == "error":
                raise ComfyError(_error_message(status))
            break
        time.sleep(2)
    else:
        raise ComfyError(f"timed out after {timeout}s waiting for ComfyUI")

    os.makedirs(out, exist_ok=True)
    saved = []
    for node_out in hist["outputs"].values():
        for img in node_out.get("images", []):
            q = urllib.parse.urlencode({"filename": img["filename"], "subfolder": img.get("subfolder", ""),
                                        "type": img.get("type", "output")})
            data = _http("GET", f"{host}/view?{q}", timeout=120).content
            dst = os.path.join(out, img["filename"])
            with open(dst, "wb") as f:
                f.write(data)
            saved.append(dst)
    print(f">> done in {time.time() - t0:.0f}s ->", ", ".join(saved))
    return saved


def submit(wf, host=DEFAULT_HOST, out="outputs", timeout=1800):
    """Queue one workflow, wait for it, download its images. Returns list of paths."""
    return wait_prompt(queue_prompt(wf, host), host, out, timeout)


def submit_many(wf, seed, count, host=DEFAULT_HOST, out="outputs", timeout=1800, progress=None):
    """Queue `count` copies of wf with seeds seed, seed+1, ... all at once (ComfyUI keeps the
    model loaded between them), then collect their images in order. progress(done, total)
    is called after each job. If some jobs fail or get cancelled, the rest are still
    collected and a ComfyError carrying the finished files is raised at the end."""
    pids = []
    for i in range(count):
        w = copy.deepcopy(wf)
        w["6"]["inputs"]["seed"] = (seed + i) % 2**32
        pids.append(queue_prompt(w, host))
    files, failed = [], {}
    for i, pid in enumerate(pids):
        try:
            files += wait_prompt(pid, host, out, timeout)
        except ComfyDown as e:
            e.files = files  # the server is gone, the remaining jobs with it
            raise
        except ComfyError as e:
            failed.setdefault(str(e), []).append(i + 1)
        if progress:
            progress(i + 1, count)
    if failed:
        lines = [f"image {', '.join(map(str, n))} of {count}: {msg}" for msg, n in failed.items()]
        raise ComfyError(f"{len(files)} of {count} image(s) finished.\n" + "\n".join(lines), files=files)
    return files


def _seed(seed):
    return random.randint(0, 2**32 - 1) if seed is None else int(seed)


def _submit(wf, seed, count, host, out, timeout, progress):
    try:
        return submit_many(wf, seed, int(count), host, out, timeout, progress)
    except ComfyError as e:
        e.seed = seed
        raise


def set_model(wf, name):
    """Point node 1 at a diffusion file: *.gguf -> UnetLoaderGGUF (leejet fork),
    anything else (fp8 / int8 / bf16 safetensors) -> stock UNETLoader."""
    if name.endswith(".gguf"):
        wf["1"] = {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": name}}
    else:
        wf["1"] = {"class_type": "UNETLoader", "inputs": {"unet_name": name, "weight_dtype": "default"}}


# ---- text to image ------------------------------------------------------------------

def run(prompt, negative="", width=1024, height=1024, steps=25, cfg=1.0, seed=None,
        gguf=None, host=DEFAULT_HOST, out="outputs", timeout=1800, clip=None, count=1,
        model=None, progress=None):
    """Returns (files, seed). count > 1 makes `count` images with seeds seed, seed+1, ...
    model: diffusion file name (*.gguf or *.safetensors); `gguf` is the old name for it."""
    with open(T2I_JSON, encoding="utf-8") as f:
        wf = json.load(f)
    seed = _seed(seed)
    wf["4"]["inputs"]["prompt"] = prompt
    wf["4"]["inputs"]["negative_prompt"] = negative
    wf["5"]["inputs"].update(width=width, height=height)
    wf["6"]["inputs"].update(steps=steps, cfg=cfg, seed=seed)
    if model or gguf:
        set_model(wf, model or gguf)
    if clip:
        wf["2"]["inputs"]["clip_name"] = clip
    print(f">> t2i seed={seed} count={count} {width}x{height} steps={steps} cfg={cfg}")
    return _submit(wf, seed, count, host, out, timeout, progress), seed


# ---- image edit ---------------------------------------------------------------------

def edit(prompt, images, negative="", resolution=0, width=None, height=None, steps=25, cfg=1.0,
         seed=None, gguf=None, clip=None, cache_device="auto", cache_dtype="default",
         host=DEFAULT_HOST, out="outputs", timeout=1800, count=1, model=None, progress=None):
    """images: list of local paths; images[0] is the edit target (canvas size), others are refs.
    resolution: pixel budget for the references (0 = keep own size, 1024 official, up to 2048).
    width/height: force the output canvas instead of following image_1 (keep it close to
    image_1's resized size or the edit shifts). count / model as in run()."""
    images = [p for p in images if p]
    if not images:
        raise ValueError("image edit needs at least one image (image_1 = edit target)")
    if len(images) > 16:
        raise ValueError("TextEncodeQwenImage21 takes at most 16 images")

    with open(EDIT_JSON, encoding="utf-8") as f:
        wf = json.load(f)
    enc = wf["4"]["inputs"]
    # drop the template's LoadImage nodes and rebuild one per supplied image
    for k in [k for k, v in wf.items() if v["class_type"] == "LoadImage"]:
        del wf[k]
    for k in [k for k in enc if k.startswith("images.")]:
        del enc[k]
    for i, path in enumerate(images, start=1):
        nid = str(100 + i)
        wf[nid] = {"class_type": "LoadImage", "inputs": {"image": upload_image(path, host)}}
        enc[f"images.image_{i}"] = [nid, 0]

    seed = _seed(seed)
    enc.update(prompt=prompt, negative_prompt=negative, resolution=int(resolution))
    wf["6"]["inputs"].update(steps=steps, cfg=cfg, seed=seed)
    wf["9"]["inputs"].update(device=cache_device, dtype=cache_dtype)
    if width and height:
        wf["5"] = {"class_type": "EmptyLatentImage",
                   "inputs": {"width": int(width), "height": int(height), "batch_size": 1}}
        wf["6"]["inputs"]["latent_image"] = ["5", 0]
    if model or gguf:
        set_model(wf, model or gguf)
    if clip:
        wf["2"]["inputs"]["clip_name"] = clip
    print(f">> edit seed={seed} count={count} images={len(images)} resolution={resolution} "
          f"canvas={'image_1' if not (width and height) else f'{width}x{height}'} steps={steps} cfg={cfg}")
    return _submit(wf, seed, count, host, out, timeout, progress), seed


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("prompt")
    ap.add_argument("--negative", default="")
    ap.add_argument("--image", action="append", default=[],
                    help="image edit mode; repeat for references (first = edit target)")
    ap.add_argument("--resolution", type=int, default=0, help="edit: reference pixel budget (0 keep, 1024, 2048)")
    ap.add_argument("--width", type=int, help="t2i: canvas width. edit: force canvas (optional)")
    ap.add_argument("--height", type=int)
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--cfg", type=float, default=1.0)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--count", type=int, default=1, help="number of images (seeds seed, seed+1, ...)")
    ap.add_argument("--model", "--gguf", dest="model",
                    help="diffusion file: qwen-image-2.1-UC-Q8_0.gguf, qwen-image-2.1-UC-fp8.safetensors, ...")
    ap.add_argument("--clip", help="override clip_name, e.g. qwen3vl_8b_bf16.safetensors")
    ap.add_argument("--cache-device", default="auto", choices=["auto", "gpu", "cpu", "off"])
    ap.add_argument("--cache-dtype", default="default", choices=["default", "int8", "int4"])
    ap.add_argument("--host", default=DEFAULT_HOST)
    ap.add_argument("--out", default="outputs")
    a = ap.parse_args()
    try:
        if a.image:
            edit(a.prompt, a.image, a.negative, a.resolution, a.width, a.height, a.steps, a.cfg, a.seed,
                 clip=a.clip, cache_device=a.cache_device, cache_dtype=a.cache_dtype, host=a.host,
                 out=a.out, count=a.count, model=a.model)
        else:
            run(a.prompt, a.negative, a.width or 1024, a.height or 1024, a.steps, a.cfg, a.seed,
                host=a.host, out=a.out, clip=a.clip, count=a.count, model=a.model)
    except ComfyError as e:
        if e.files:
            print(">> saved anyway:", ", ".join(e.files))
        sys.exit(str(e))
