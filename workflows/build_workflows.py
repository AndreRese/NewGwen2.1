#!/usr/bin/env python
"""Generate the t2i and image-edit workflows in ComfyUI UI format and API format.

    python build_workflows.py [--quant Q4_K_M] [--text-encoder qwen3vl_8b_int8_convrot.safetensors]
                              [--safetensors qwen-image-2.1-UC-int8_convrot.safetensors]

The graphs mirror Comfy-Org's official image_qwen_image_2_1_t2i / _image_edit templates
(which hide everything in a subgraph) but flattened. The *_gguf_* files use UnetLoaderGGUF
(leejet fork) with the uncensored qwen-image-2.1-UC-*.gguf; the *_safetensors_* files keep
the stock UNETLoader for the fp8 / int8_convrot / base bf16 safetensors. generate.py only
needs the GGUF API files and swaps node 1 by file extension. Edit graph: LoadImage(s) -> TextEncodeQwenImage21
(images.image_N + vae); its `latent` output (canvas = image_1 size) feeds the KSampler;
QwenImage21Cache sits between the loader and the sampler to set KV-cache device / precision.
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
- Loader: `Unet Loader (GGUF)` from the **leejet** ComfyUI-GGUF fork for *.gguf; the stock `Load Diffusion Model` (UNETLoader) for *.safetensors (UC fp8 / int8_convrot, base bf16).
- Files: abenzerps/Qwen-Image-2.1-Uncensored-GGUF (`UC` = uncensored; base quants on its `base` branch) and Comfy-Org/Qwen-Image-2.1 (unquantized base) on HF.
"""


def loader_api(name):
    """Node 1 in API format: GGUF loader for *.gguf, stock UNETLoader for safetensors."""
    if name.endswith(".gguf"):
        return {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": name}}
    return {"class_type": "UNETLoader", "inputs": {"unet_name": name, "weight_dtype": "default"}}


def loader_ui(name, node, inp, out, l_model, pos=(40, 60)):
    """Node 1 in UI format (same helpers as the graph builders)."""
    if name.endswith(".gguf"):
        return node(1, "UnetLoaderGGUF", list(pos), [380, 60],
                    [inp("unet_name", "COMBO", widget={"name": "unet_name"})],
                    [out("MODEL", "MODEL", [l_model])], [name], "Unet Loader (GGUF)")
    return node(1, "UNETLoader", list(pos), [380, 90],
                [inp("unet_name", "COMBO", widget={"name": "unet_name"}),
                 inp("weight_dtype", "COMBO", widget={"name": "weight_dtype"})],
                [out("MODEL", "MODEL", [l_model])], [name, "default"], "Load Diffusion Model")


def build(gguf, text_encoder, vae, width=1024, height=1024, steps=25, cfg=1.0):
    # ---- API format (what POST /prompt expects) ----------------------------------------
    api = {
        "1": loader_api(gguf),
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
        loader_ui(gguf, node, inp, out, l_model),
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
    ui = {"id": f"qwen-image-2.1-{'gguf' if gguf.endswith('.gguf') else 'safetensors'}-t2i", "revision": 0, "last_node_id": 9,
          "last_link_id": len(links), "nodes": nodes, "links": links, "groups": [],
          "config": {}, "extra": {"ds": {"scale": 0.8, "offset": [0, 0]}}, "version": 0.4}
    return ui, api


EDIT_PROMPT = "Keep the character and pose in <image1> unchanged, change the background to a sunny beach."

EDIT_NOTE = """## Qwen-Image 2.1 — Image edit (GGUF)

- **image_1 is the edit target**; image_2… are references. Mention them in the prompt as `<image1>`, `<image2>`, …
- **Canvas = image_1 size** (the `latent` output of Text Encode). To force another size, feed an `EmptyLatentImage` close to image_1's resized size instead, or the edit shifts.
- **resolution** = total pixel budget for the references, aspect kept, multiples of 32. `0` = keep each image's own size; official default 1024; up to 2048.
- Up to 16 reference inputs; add `LoadImage` nodes and drag them into `images.image_N`.
- `Qwen Image 2.1 Cache`: `dtype int8` halves the KV cache (edits on tight VRAM); `int4` quarters it, less accurate.
- cfg 1.0 / euler / simple / 25 steps as for t2i.
"""


def build_edit(gguf, text_encoder, vae, steps=25, cfg=1.0, resolution=0, n_images=2,
               cache_device="auto", cache_dtype="default"):
    # ---- API format --------------------------------------------------------------------
    enc = {"clip": ["2", 0], "vae": ["3", 0], "prompt": EDIT_PROMPT, "negative_prompt": "",
           "resolution": resolution}
    api = {
        "1": loader_api(gguf),
        "2": {"class_type": "CLIPLoader",
              "inputs": {"clip_name": text_encoder, "type": "qwen_image", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": vae}},
        "4": {"class_type": "TextEncodeQwenImage21", "inputs": enc},
        "6": {"class_type": "KSampler",
              "inputs": {"model": ["9", 0], "positive": ["4", 0], "negative": ["4", 1],
                         "latent_image": ["4", 2], "seed": 0, "steps": steps, "cfg": cfg,
                         "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}},
        "7": {"class_type": "VAEDecode", "inputs": {"samples": ["6", 0], "vae": ["3", 0]}},
        "8": {"class_type": "SaveImage", "inputs": {"images": ["7", 0], "filename_prefix": "Qwen_image_2.1_edit"}},
        "9": {"class_type": "QwenImage21Cache",
              "inputs": {"model": ["1", 0], "device": cache_device, "dtype": cache_dtype}},
    }
    for i in range(1, n_images + 1):
        api[str(9 + i)] = {"class_type": "LoadImage", "inputs": {"image": f"image_{i}.png"}}
        enc[f"images.image_{i}"] = [str(9 + i), 0]

    # ---- UI format ---------------------------------------------------------------------
    links = []

    def link(a, aslot, b, bslot, t):
        links.append([len(links) + 1, a, aslot, b, bslot, t])
        return len(links)

    def node(id_, type_, pos, size, inputs, outputs, widgets, title=None, color=None):
        n = {"id": id_, "type": type_, "pos": pos, "size": size, "flags": {}, "order": id_ - 1,
             "mode": 0, "inputs": inputs, "outputs": outputs, "properties": {"Node name for S&R": type_},
             "widgets_values": widgets}
        if title:
            n["title"] = title
        if color:
            n["color"], n["bgcolor"] = color
        return n

    def inp(name, t, l=None, **kw):
        return {"name": name, "type": t, "link": l, **kw}

    def out(name, t, ls):
        return {"name": name, "type": t, "links": ls, "slot_index": 0}

    l_model = link(1, 0, 9, 0, "MODEL")
    l_cached = link(9, 0, 6, 0, "MODEL")
    l_clip = link(2, 0, 4, 0, "CLIP")
    l_vae_enc = link(3, 0, 4, 1, "VAE")
    l_vae_dec = link(3, 0, 7, 1, "VAE")
    l_pos = link(4, 0, 6, 1, "CONDITIONING")
    l_neg = link(4, 1, 6, 2, "CONDITIONING")
    l_lat = link(4, 2, 6, 3, "LATENT")
    l_samp = link(6, 0, 7, 0, "LATENT")
    l_img = link(7, 0, 8, 0, "IMAGE")
    l_imgs = [link(9 + i, 0, 4, 4 + i, "IMAGE") for i in range(1, n_images + 1)]

    enc_inputs = [inp("clip", "CLIP", l_clip), inp("vae", "VAE", l_vae_enc, shape=7),
                  inp("prompt", "STRING", None, widget={"name": "prompt"}),
                  inp("negative_prompt", "STRING", None, widget={"name": "negative_prompt"}),
                  inp("resolution", "INT", None, widget={"name": "resolution"})]
    enc_inputs += [inp(f"images.image_{i}", "IMAGE", l_imgs[i - 1], shape=7) for i in range(1, n_images + 1)]
    enc_inputs.append(inp(f"images.image_{n_images + 1}", "IMAGE", None, shape=7))

    nodes = [
        loader_ui(gguf, node, inp, out, l_model),
        node(9, "QwenImage21Cache", [40, 170], [380, 90],
             [inp("model", "MODEL", l_model), inp("device", "COMBO", None, widget={"name": "device"}),
              inp("dtype", "COMBO", None, widget={"name": "dtype"})],
             [out("MODEL", "MODEL", [l_cached])], [cache_device, cache_dtype]),
        node(2, "CLIPLoader", [40, 310], [380, 110],
             [inp("clip_name", "COMBO", widget={"name": "clip_name"}),
              inp("type", "COMBO", widget={"name": "type"}),
              inp("device", "COMBO", widget={"name": "device"})],
             [out("CLIP", "CLIP", [l_clip])], [text_encoder, "qwen_image", "default"]),
        node(3, "VAELoader", [40, 470], [380, 60],
             [inp("vae_name", "COMBO", widget={"name": "vae_name"})],
             [out("VAE", "VAE", [l_vae_enc, l_vae_dec])], [vae]),
        node(4, "TextEncodeQwenImage21", [900, 60], [460, 420], enc_inputs,
             [out("positive", "CONDITIONING", [l_pos]), out("negative", "CONDITIONING", [l_neg]),
              out("latent", "LATENT", [l_lat])],
             [EDIT_PROMPT, "", resolution]),
        node(6, "KSampler", [1410, 60], [320, 280],
             [inp("model", "MODEL", l_cached), inp("positive", "CONDITIONING", l_pos),
              inp("negative", "CONDITIONING", l_neg), inp("latent_image", "LATENT", l_lat),
              inp("seed", "INT", None, widget={"name": "seed"}),
              inp("steps", "INT", None, widget={"name": "steps"}),
              inp("cfg", "FLOAT", None, widget={"name": "cfg"}),
              inp("sampler_name", "COMBO", None, widget={"name": "sampler_name"}),
              inp("scheduler", "COMBO", None, widget={"name": "scheduler"}),
              inp("denoise", "FLOAT", None, widget={"name": "denoise"})],
             [out("LATENT", "LATENT", [l_samp])],
             [0, "randomize", steps, cfg, "euler", "simple", 1.0]),
        node(7, "VAEDecode", [1780, 60], [210, 50],
             [inp("samples", "LATENT", l_samp), inp("vae", "VAE", l_vae_dec)],
             [out("IMAGE", "IMAGE", [l_img])], None),
        node(8, "SaveImage", [1780, 160], [420, 400],
             [inp("images", "IMAGE", l_img),
              inp("filename_prefix", "STRING", None, widget={"name": "filename_prefix"})],
             [], ["Qwen_image_2.1_edit"]),
        {"id": 5, "type": "MarkdownNote", "pos": [40, 570], "size": [380, 300],
         "flags": {}, "order": 4, "mode": 0, "inputs": [], "outputs": [], "properties": {},
         "widgets_values": [EDIT_NOTE], "color": "#432", "bgcolor": "#653"},
    ]
    for i in range(1, n_images + 1):
        nodes.append(node(9 + i, "LoadImage", [470, 60 + (i - 1) * 380], [380, 340],
                          [inp("image", "COMBO", widget={"name": "image"}),
                           inp("upload", "IMAGEUPLOAD", widget={"name": "upload"})],
                          [out("IMAGE", "IMAGE", [l_imgs[i - 1]]), out("MASK", "MASK", None)],
                          [f"image_{i}.png", "image"],
                          "image_1 (edit target)" if i == 1 else f"image_{i} (reference)",
                          ("#232", "#353") if i == 1 else None))
    ui = {"id": f"qwen-image-2.1-{'gguf' if gguf.endswith('.gguf') else 'safetensors'}-edit", "revision": 0, "last_node_id": 9 + n_images,
          "last_link_id": len(links), "nodes": nodes, "links": links, "groups": [],
          "config": {}, "extra": {"ds": {"scale": 0.7, "offset": [0, 0]}}, "version": 0.4}
    return ui, api


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quant", default="Q4_K_M")
    ap.add_argument("--text-encoder", default="qwen3vl_8b_int8_convrot.safetensors")
    ap.add_argument("--vae", default="qwen_image_2.1_vae_bf16.safetensors")
    ap.add_argument("--safetensors", default="qwen-image-2.1-UC-int8_convrot.safetensors",
                    help="file for the *_safetensors_* canvas workflows (UNETLoader)")
    a = ap.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    gguf = f"qwen-image-2.1-UC-{a.quant}.gguf"
    ui, api = build(gguf, a.text_encoder, a.vae)
    ui_e, api_e = build_edit(gguf, a.text_encoder, a.vae)
    ui_f, _ = build(a.safetensors, a.text_encoder, a.vae)
    ui_fe, _ = build_edit(a.safetensors, a.text_encoder, a.vae)
    for name, data in (("qwen_image_2.1_gguf_t2i.json", ui), ("qwen_image_2.1_gguf_t2i_api.json", api),
                       ("qwen_image_2.1_gguf_edit.json", ui_e), ("qwen_image_2.1_gguf_edit_api.json", api_e),
                       ("qwen_image_2.1_safetensors_t2i.json", ui_f),
                       ("qwen_image_2.1_safetensors_edit.json", ui_fe)):
        with open(os.path.join(here, name), "w", encoding="utf-8", newline=chr(10)) as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print("wrote", name)
