#!/usr/bin/env python3
"""Copy a ComfyUI diffusion model and mark attention modules for comfy-kitchen int8 attention: adds
"<prefix>.comfy_attention.config" = {"attention": "comfy_kitchen_int8"} for every module prefix matching REGEX.

usage: add_int8_attention.py IN.safetensors OUT.safetensors --attention REGEX
  e.g. --attention 'transformer_blocks\\.\\d+\\.attn1$'     (LTX-2.5: self-attention of every block)

Only for int8 ConvRot checkpoints (the official *_int8_convrot files) and a comfy-kitchen build with int8 attention.
The whole file is loaded into RAM (about the size of the checkpoint) and written again next to it.
"""
import argparse
import json
import re

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--attention", required=True)
    a = ap.parse_args()
    pat = re.compile(a.attention)
    out, prefixes = {}, set()
    with safe_open(a.src, "pt", device="cpu") as f:
        meta = f.metadata()
        for k in f.keys():
            out[k] = f.get_tensor(k)
            for suffix in (".to_q.weight",):
                if k.endswith(suffix) and pat.search(k[:-len(suffix)]):
                    prefixes.add(k[:-len(suffix)])
    cfg = torch.tensor(list(json.dumps({"attention": "comfy_kitchen_int8"}).encode()), dtype=torch.uint8)
    for p in sorted(prefixes):
        out[p + ".comfy_attention.config"] = cfg.clone()
    save_file(out, a.dst, metadata=meta)
    print(f"int8 attention on {len(prefixes)} modules -> {a.dst}")


if __name__ == "__main__":
    main()
