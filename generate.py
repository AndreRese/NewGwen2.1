#!/usr/bin/env python
"""Headless generation through a running ComfyUI (no browser needed).

Text to image:
    python generate.py "a cat wearing a spacesuit, studio photo"
    python generate.py "..." --width 2048 --height 2048 --steps 40 --seed 42 --out outputs/

Image edit (image_1 is the edit target, the rest are references; use <image1>, <image2> in the prompt):
    python generate.py "Keep <image1> unchanged, put the shirt from <image2> on her" --image a.png --image b.png
    python generate.py "make it night" --image a.png --resolution 1024 --cache-dtype int8

Loads workflows/qwen_image_2.1_gguf_{t2i,edit}_api.json, patches the inputs, POSTs to
/prompt, waits for the job and copies the PNGs to --out.
"""
import argparse
import json
import os
import random
import sys
import time
import urllib.parse

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
T2I_JSON = os.path.join(HERE, "workflows", "qwen_image_2.1_gguf_t2i_api.json")
EDIT_JSON = os.path.join(HERE, "workflows", "qwen_image_2.1_gguf_edit_api.json")
DEFAULT_HOST = "http://127.0.0.1:8188"


# ---- ComfyUI API plumbing -----------------------------------------------------------

def upload_image(path, host=DEFAULT_HOST):
    """Upload a local image to ComfyUI's input folder; returns the name LoadImage expects."""
    with open(path, "rb") as f:
        r = requests.post(f"{host}/upload/image",
                          files={"image": (os.path.basename(path), f)},
                          data={"overwrite": "true", "type": "input", "subfolder": "qwen21"})
    r.raise_for_status()
    j = r.json()
    return f"{j['subfolder']}/{j['name']}" if j.get("subfolder") else j["name"]


def submit(wf, host=DEFAULT_HOST, out="outputs", timeout=1800):
    """Queue an API-format workflow, wait for it, download its images. Returns list of paths."""
    r = requests.post(f"{host}/prompt", json={"prompt": wf})
    if r.status_code != 200:
        sys.exit(f"ComfyUI rejected the prompt ({r.status_code}):\n{r.text[:2000]}")
    pid = r.json()["prompt_id"]
    print(f">> queued {pid}")

    t0 = time.time()
    while time.time() - t0 < timeout:
        hist = requests.get(f"{host}/history/{pid}").json().get(pid)
        if hist:
            status = hist.get("status", {})
            if status.get("status_str") == "error":
                sys.exit("ComfyUI reported an error:\n" + json.dumps(status, indent=2)[:2000])
            break
        time.sleep(2)
    else:
        sys.exit("timed out waiting for ComfyUI")

    os.makedirs(out, exist_ok=True)
    saved = []
    for node_out in hist["outputs"].values():
        for img in node_out.get("images", []):
            q = urllib.parse.urlencode({"filename": img["filename"], "subfolder": img.get("subfolder", ""),
                                        "type": img.get("type", "output")})
            data = requests.get(f"{host}/view?{q}").content
            dst = os.path.join(out, img["filename"])
            with open(dst, "wb") as f:
                f.write(data)
            saved.append(dst)
    print(f">> done in {time.time() - t0:.0f}s ->", ", ".join(saved))
    return saved


def _seed(seed):
    return random.randint(0, 2**32 - 1) if seed is None else int(seed)


# ---- text to image ------------------------------------------------------------------

def run(prompt, negative="", width=1024, height=1024, steps=25, cfg=1.0, seed=None,
        gguf=None, host=DEFAULT_HOST, out="outputs", timeout=1800, clip=None):
    with open(T2I_JSON, encoding="utf-8") as f:
        wf = json.load(f)
    seed = _seed(seed)
    wf["4"]["inputs"]["prompt"] = prompt
    wf["4"]["inputs"]["negative_prompt"] = negative
    wf["5"]["inputs"].update(width=width, height=height)
    wf["6"]["inputs"].update(steps=steps, cfg=cfg, seed=seed)
    if gguf:
        wf["1"]["inputs"]["unet_name"] = gguf
    if clip:
        wf["2"]["inputs"]["clip_name"] = clip
    print(f">> t2i seed={seed} {width}x{height} steps={steps} cfg={cfg}")
    return submit(wf, host, out, timeout), seed


# ---- image edit ---------------------------------------------------------------------

def edit(prompt, images, negative="", resolution=0, width=None, height=None, steps=25, cfg=1.0,
         seed=None, gguf=None, clip=None, cache_device="auto", cache_dtype="default",
         host=DEFAULT_HOST, out="outputs", timeout=1800):
    """images: list of local paths; images[0] is the edit target (canvas size), others are refs.
    resolution: pixel budget for the references (0 = keep own size, 1024 official, up to 2048).
    width/height: force the output canvas instead of following image_1 (keep it close to
    image_1's resized size or the edit shifts)."""
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
    if gguf:
        wf["1"]["inputs"]["unet_name"] = gguf
    if clip:
        wf["2"]["inputs"]["clip_name"] = clip
    print(f">> edit seed={seed} images={len(images)} resolution={resolution} "
          f"canvas={'image_1' if not (width and height) else f'{width}x{height}'} steps={steps} cfg={cfg}")
    return submit(wf, host, out, timeout), seed


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
    ap.add_argument("--gguf", help="override unet_name, e.g. qwen-image-2.1-Q8_0.gguf")
    ap.add_argument("--clip", help="override clip_name, e.g. qwen3vl_8b_bf16.safetensors")
    ap.add_argument("--cache-device", default="auto", choices=["auto", "gpu", "cpu", "off"])
    ap.add_argument("--cache-dtype", default="default", choices=["default", "int8", "int4"])
    ap.add_argument("--host", default=DEFAULT_HOST)
    ap.add_argument("--out", default="outputs")
    a = ap.parse_args()
    if a.image:
        edit(a.prompt, a.image, a.negative, a.resolution, a.width, a.height, a.steps, a.cfg, a.seed,
             a.gguf, a.clip, a.cache_device, a.cache_dtype, a.host, a.out)
    else:
        run(a.prompt, a.negative, a.width or 1024, a.height or 1024, a.steps, a.cfg, a.seed,
            a.gguf, a.host, a.out, clip=a.clip)
