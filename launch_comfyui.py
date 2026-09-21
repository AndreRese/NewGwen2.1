#!/usr/bin/env python
"""Start ComfyUI in the background and expose it from Colab.

    python launch_comfyui.py                 # Colab proxy URL (no external tunnel)
    python launch_comfyui.py --tunnel        # also open a cloudflared quick tunnel
    python launch_comfyui.py --lowvram       # T4 / 15 GB cards

Run it from a notebook cell with `!python launch_comfyui.py`, or import and call
`launch()` from the notebook so the proxy link renders as a clickable link.
"""
import argparse
import os
import re
import shutil
import socket
import subprocess
import sys
import time
import urllib.request

PORT = 8188
LOG = "comfyui.log"
URLS = {"proxy": None, "tunnel": None}  # filled by launch(), read by app_gradio


def port_open(port):
    with socket.socket() as s:
        return s.connect_ex(("127.0.0.1", port)) == 0


def wait_for(port, timeout=600):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if port_open(port):
            return True
        time.sleep(2)
    return False


def colab_proxy_url(port):
    try:
        from google.colab.output import eval_js  # noqa
        return eval_js(f"google.colab.kernel.proxyPort({port})")
    except Exception:
        return None


def cloudflared(port):
    if not shutil.which("cloudflared"):
        print(">> installing cloudflared")
        deb = "/tmp/cloudflared.deb"
        urllib.request.urlretrieve(
            "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb", deb)
        subprocess.run(["dpkg", "-i", deb], check=True, stdout=subprocess.DEVNULL)
    p = subprocess.Popen(["cloudflared", "tunnel", "--url", f"http://127.0.0.1:{port}"],
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in p.stdout:
        m = re.search(r"https://[-a-z0-9]+\.trycloudflare\.com", line)
        if m:
            return m.group(0)
    return None


def launch(comfy_dir="ComfyUI", lowvram=False, tunnel=False, extra=()):
    if port_open(PORT):
        print(f">> ComfyUI already listening on :{PORT}")
    else:
        cmd = [sys.executable, "main.py", "--listen", "127.0.0.1", "--port", str(PORT),
               "--dont-print-server", "--disable-auto-launch", "--preview-method", "auto"]
        if lowvram:
            cmd.append("--lowvram")
        cmd += list(extra)
        print(">>", " ".join(cmd))
        log = open(os.path.join(comfy_dir, LOG), "w")
        subprocess.Popen(cmd, cwd=comfy_dir, stdout=log, stderr=subprocess.STDOUT)
        if not wait_for(PORT):
            sys.exit(f"ComfyUI did not start — check {comfy_dir}/{LOG}")

    url = URLS["proxy"] = colab_proxy_url(PORT)
    if url:
        print(f"\n>> ComfyUI (Colab proxy): {url}\n")
    else:
        print(f"\n>> ComfyUI: http://127.0.0.1:{PORT}\n")
    if tunnel or not url:
        t = URLS["tunnel"] = cloudflared(PORT)
        print(f">> ComfyUI (cloudflare):   {t}\n" if t else ">> cloudflared failed to give a URL")
    return url


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--comfy-dir", default="ComfyUI")
    ap.add_argument("--lowvram", action="store_true")
    ap.add_argument("--tunnel", action="store_true", help="also open a cloudflared tunnel")
    args, extra = ap.parse_known_args()
    launch(args.comfy_dir, args.lowvram, args.tunnel, extra)
