#!/usr/bin/env python
"""Start ComfyUI in the background and expose it from Colab.

    python launch_comfyui.py                 # Colab proxy URL (no external tunnel)
    python launch_comfyui.py --tunnel        # also open a cloudflared quick tunnel
    python launch_comfyui.py --lowvram       # T4 / 15 GB cards
    python launch_comfyui.py --restart       # reload after updating / patching nodes

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


def log_tail(comfy_dir="ComfyUI", n=40):
    try:
        with open(os.path.join(comfy_dir, LOG), encoding="utf-8", errors="replace") as f:
            return "".join(f.readlines()[-n:])
    except OSError:
        return ""


def wait_for(port, timeout=600, proc=None, comfy_dir="ComfyUI"):
    """Poll the port; bail out at once if the server process dies, and print a heartbeat
    with the last log line every 20 s so a slow start is distinguishable from a hang."""
    t0 = time.time()
    last_beat = t0
    while time.time() - t0 < timeout:
        if port_open(port):
            return True
        if proc is not None and proc.poll() is not None:
            print(f">> ComfyUI exited with code {proc.returncode} during startup. Last log lines:\n")
            print(log_tail(comfy_dir))
            return False
        if time.time() - last_beat >= 20:
            last_beat = time.time()
            tail = log_tail(comfy_dir, 1).strip()
            print(f">> starting ComfyUI... {last_beat - t0:.0f}s  |  {tail[:160]}")
        time.sleep(2)
    print(f">> ComfyUI did not open :{port} within {timeout}s. Last log lines:\n")
    print(log_tail(comfy_dir))
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


def stop():
    """Kill whatever is listening on the ComfyUI port (Linux/Colab)."""
    subprocess.run(["fuser", "-k", f"{PORT}/tcp"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(15):
        if not port_open(PORT):
            return True
        time.sleep(1)
    return False


def launch(comfy_dir="ComfyUI", lowvram=False, tunnel=False, extra=(), restart=False):
    if restart and port_open(PORT):
        print(">> stopping running ComfyUI")
        stop()
    if port_open(PORT):
        print(f">> ComfyUI already listening on :{PORT}")
    else:
        cmd = [sys.executable, "main.py", "--listen", "127.0.0.1", "--port", str(PORT),
               "--dont-print-server", "--disable-auto-launch", "--preview-method", "auto"]
        if lowvram:
            cmd.append("--lowvram")
        cmd += list(extra)
        print(">>", " ".join(cmd))
        if not os.path.isfile(os.path.join(comfy_dir, "main.py")):
            sys.exit(f"{comfy_dir}/main.py not found — run setup_colab.sh first (cwd: {os.getcwd()})")
        log = open(os.path.join(comfy_dir, LOG), "w")
        proc = subprocess.Popen(cmd, cwd=comfy_dir, stdout=log, stderr=subprocess.STDOUT)
        if not wait_for(PORT, proc=proc, comfy_dir=comfy_dir):
            sys.exit(f"ComfyUI did not start — full log: {comfy_dir}/{LOG}")

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
    ap.add_argument("--restart", action="store_true", help="kill a running ComfyUI first (after patches/updates)")
    args, extra = ap.parse_known_args()
    launch(args.comfy_dir, args.lowvram, args.tunnel, extra, restart=args.restart)
