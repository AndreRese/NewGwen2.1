#!/usr/bin/env python
"""Generate the t2i workflow in ComfyUI UI format and API format.

    python build_workflows.py [--quant Q4_K_M] [--text-encoder qwen3vl_8b_int8_convrot.safetensors]

The graph mirrors Comfy-Org's official image_qwen_image_2_1_t2i template (which hides
everything in a subgraph) but flattened and with UNETLoader swapped for UnetLoaderGGUF.
"""
import argparse
import json
import os

PROMPT = ("Cinematic photo of a woman in a red dress standing on a rooftop at dusk, "
          "city lights bokeh behind her, 85mm lens, shallow depth of field, film grain.")

NOTE = """## Qwen-Image 2.1 Uncensored (GGUF)

- **cfg 1.0** is the official Qwen-Image 2.1 path; the negative prompt is ignored at cfg 1. Raise cfg only if you use a negative.
- **steps**: 25 is fast; the official pipeline uses 40-50 with euler.
- Native 2K: set 2048x2048 on EmptyLatentImage (multiples of 32).
- Text encoder: `CLIPLoader` type must be `qwen_image`.
- Loader: `Unet Loader (GGUF)` from the **leejet** ComfyUI-GGUF fork.
- Files: abenzerps/Qwen-Image-2.1-Uncensored-GGUF on HF.
"""


def build(gguf, text_encoder, vae, width=1024, height=1024, steps=25, cfg=1.0):
    # ---- API format (what POST /prompt expects) ----------------------------------------
    api = {
        "1": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": gguf}},
        "2": {"class_type": "CLIPLoader",
              "inputs": {"clip_name": text_encoder, "type": "qwen_image", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": vae}},
        "4": {"class_type": "TextEncodeQwenImage21",
              "inputs": {"clip": ["2", 0], "prompt": PROMPT, "negative_prompt": "", "resolution": 1024}},
        "5": {"class_type": "EmptyLatentImage",
              "inputs": {"width": width, "height": height, "batch_size": 1}},
        "6": {"class_type": "KSampler",
              "inputs": {"model": ["1", 0], "positive": ["4", 0], "negative": ["4", 1],
                         "latent_image": ["5", 0], "seed": 0, "steps": steps, "cfg": cfg,
                         "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}},
        "7": {"class_type": "VAEDecode", "inputs": {"samples": ["6", 0], "vae": ["3", 0]}},
        "8": {"class_type": "SaveImage", "inputs": {"images": ["7", 0], "filename_prefix": "Qwen_image_2.1"}},
    }

    # ---- UI format (LiteGraph) ---------------------------------------------------------
    links = []  # [id, from_node, from_slot, to_node, to_slot, type]

    def link(a, aslot, b, bslot, t):
        links.append([len(links) + 1, a, aslot, b, bslot, t])
        return len(links)

    l_model = link(1, 0, 6, 0, "MODEL")
    l_clip = link(2, 0, 4, 0, "CLIP")
    l_vae = link(3, 0, 7, 1, "VAE")
    l_pos = link(4, 0, 6, 1, "CONDITIONING")
    l_neg = link(4, 1, 6, 2, "CONDITIONING")
    l_lat = link(5, 0, 6, 3, "LATENT")
    l_samp = link(6, 0, 7, 0, "LATENT")
    l_img = link(7, 0, 8, 0, "IMAGE")

    def node(id_, type_, pos, size, inputs, outputs, widgets, title=None):
        n = {"id": id_, "type": type_, "pos": pos, "size": size, "flags": {}, "order": id_ - 1,
             "mode": 0, "inputs": inputs, "outputs": outputs, "properties": {"Node name for S&R": type_},
             "widgets_values": widgets}
        if title:
            n["title"] = title
        return n

    def inp(name, t, l=None, **kw):
        return {"name": name, "type": t, "link": l, **kw}

    def out(name, t, ls):
        return {"name": name, "type": t, "links": ls, "slot_index": 0}

    nodes = [
        node(1, "UnetLoaderGGUF", [40, 60], [380, 60],
             [inp("unet_name", "COMBO", widget={"name": "unet_name"})],
             [out("MODEL", "MODEL", [l_model])], [gguf], "Unet Loader (GGUF)"),
        node(2, "CLIPLoader", [40, 180], [380, 110],
             [inp("clip_name", "COMBO", widget={"name": "clip_name"}),
              inp("type", "COMBO", widget={"name": "type"}),
              inp("device", "COMBO", widget={"name": "device"})],
             [out("CLIP", "CLIP", [l_clip])], [text_encoder, "qwen_image", "default"]),
        node(3, "VAELoader", [40, 340], [380, 60],
             [inp("vae_name", "COMBO", widget={"name": "vae_name"})],
             [out("VAE", "VAE", [l_vae])], [vae]),
        node(4, "TextEncodeQwenImage21", [470, 60], [460, 340],
             [inp("clip", "CLIP", l_clip),
              inp("vae", "VAE", None, shape=7),
              inp("prompt", "STRING", None, widget={"name": "prompt"}),
              inp("negative_prompt", "STRING", None, widget={"name": "negative_prompt"}),
              inp("resolution", "INT", None, widget={"name": "resolution"}),
              inp("images.image_1", "IMAGE", None, shape=7)],
             [out("positive", "CONDITIONING", [l_pos]), out("negative", "CONDITIONING", [l_neg]),
              out("latent", "LATENT", None)],
             [PROMPT, "", 1024]),
        node(5, "EmptyLatentImage", [470, 450], [300, 110],
             [inp("width", "INT", None, widget={"name": "width"}),
              inp("height", "INT", None, widget={"name": "height"}),
              inp("batch_size", "INT", None, widget={"name": "batch_size"})],
             [out("LATENT", "LATENT", [l_lat])], [width, height, 1]),
        node(6, "KSampler", [980, 60], [320, 280],
             [inp("model", "MODEL", l_model), inp("positive", "CONDITIONING", l_pos),
              inp("negative", "CONDITIONING", l_neg), inp("latent_image", "LATENT", l_lat),
              inp("seed", "INT", None, widget={"name": "seed"}),
              inp("steps", "INT", None, widget={"name": "steps"}),
              inp("cfg", "FLOAT", None, widget={"name": "cfg"}),
              inp("sampler_name", "COMBO", None, widget={"name": "sampler_name"}),
              inp("scheduler", "COMBO", None, widget={"name": "scheduler"}),
              inp("denoise", "FLOAT", None, widget={"name": "denoise"})],
             [out("LATENT", "LATENT", [l_samp])],
             [0, "randomize", steps, cfg, "euler", "simple", 1.0]),
        node(7, "VAEDecode", [1350, 60], [210, 50],
             [inp("samples", "LATENT", l_samp), inp("vae", "VAE", l_vae)],
             [out("IMAGE", "IMAGE", [l_img])], None),
        node(8, "SaveImage", [1350, 160], [420, 400],
             [inp("images", "IMAGE", l_img),
              inp("filename_prefix", "STRING", None, widget={"name": "filename_prefix"})],
             [], ["Qwen_image_2.1"]),
        {"id": 9, "type": "MarkdownNote", "pos": [40, 460], "size": [380, 260],
         "flags": {}, "order": 8, "mode": 0, "inputs": [], "outputs": [], "properties": {},
         "widgets_values": [NOTE], "color": "#432", "bgcolor": "#653"},
    ]
    ui = {"id": "qwen-image-2.1-uncensored-gguf-t2i", "revision": 0, "last_node_id": 9,
          "last_link_id": len(links), "nodes": nodes, "links": links, "groups": [],
          "config": {}, "extra": {"ds": {"scale": 0.8, "offset": [0, 0]}}, "version": 0.4}
    return ui, api


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quant", default="Q4_K_M")
    ap.add_argument("--text-encoder", default="qwen3vl_8b_int8_convrot.safetensors")
    ap.add_argument("--vae", default="qwen_image_2.1_vae_bf16.safetensors")
    a = ap.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    ui, api = build(f"qwen-image-2.1-{a.quant}.gguf", a.text_encoder, a.vae)
    for name, data in (("qwen_image_2.1_gguf_t2i.json", ui), ("qwen_image_2.1_gguf_t2i_api.json", api)):
        with open(os.path.join(here, name), "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print("wrote", name)
