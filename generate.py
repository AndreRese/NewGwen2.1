#!/usr/bin/env python
"""Headless generation through a running ComfyUI (no browser needed).

    python generate.py "a cat wearing a spacesuit, studio photo"
    python generate.py "..." --width 2048 --height 2048 --steps 40 --seed 42 --out outputs/

Loads workflows/qwen_image_2.1_gguf_t2i_api.json, patches prompt / size / steps / seed,
POSTs it to /prompt, waits for the job and copies the PNGs to --out.
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
API_JSON = os.path.join(HERE, "workflows", "qwen_image_2.1_gguf_t2i_api.json")


def run(prompt, negative="", width=1024, height=1024, steps=25, cfg=1.0, seed=None,
        gguf=None, host="http://127.0.0.1:8188", out="outputs", timeout=1800):
    with open(API_JSON, encoding="utf-8") as f:
        wf = json.load(f)

    seed = random.randint(0, 2**32 - 1) if seed is None else seed
    wf["4"]["inputs"]["prompt"] = prompt
    wf["4"]["inputs"]["negative_prompt"] = negative
    wf["5"]["inputs"].update(width=width, height=height)
    wf["6"]["inputs"].update(steps=steps, cfg=cfg, seed=seed)
    if gguf:
        wf["1"]["inputs"]["unet_name"] = gguf

    r = requests.post(f"{host}/prompt", json={"prompt": wf})
    r.raise_for_status()
    pid = r.json()["prompt_id"]
    print(f">> queued {pid}  seed={seed}  {width}x{height}  steps={steps} cfg={cfg}")

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


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt")
    ap.add_argument("--negative", default="")
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--cfg", type=float, default=1.0)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--gguf", help="override unet_name, e.g. qwen-image-2.1-Q8_0.gguf")
    ap.add_argument("--host", default="http://127.0.0.1:8188")
    ap.add_argument("--out", default="outputs")
    a = ap.parse_args()
    run(a.prompt, a.negative, a.width, a.height, a.steps, a.cfg, a.seed, a.gguf, a.host, a.out)
