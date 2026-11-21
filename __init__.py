"""Speed-ups for ComfyUI on AMD RDNA4 (R9700, gfx1201) as opt-in nodes, leaving ComfyUI core files untouched.

"Text Encoder BF16" (class MiniMaxH3TextEncoderBF16, name kept so saved workflows still load): put it after the CLIP loader. CLIPs that pass through it run the language model of
the MiniMax H3 text encoder (Qwen3-VL-32B) in bf16; every other CLIP is unchanged. ComfyUI's SDClipModel calls text
encoders with dtype=float32, so the hidden states stay fp32 and each nvfp4 layer is dequantized to fp32 for an fp32
GEMM. On ROCm (R9700) that lands on a small-tile fp32 kernel: ~32 s of a ~38 s encode. The weights file, the loader and
the vision tower are unchanged; the conditioning still leaves the encoder as fp32. The MiniMax H3 Image encoder
(PH3ImageQwen3VL) gets the same bf16 language model; its hidden states go back to fp32 before its fp32 text connector.
Plain Qwen3-VL encoders (Qwen-Image 2.1's qwen3vl_8b, Krea 2's qwen3vl_4b) and Qwen3.5 (MOSS Image's qwen35_9b) get
a bf16 language model and a bf16 vision tower, and their outputs go back to fp32.

"Unpinned Offload": for the MODEL and/or CLIP passed through it, weights moved out of VRAM to make room for another
model are not page-locked (cudaHostRegister) in RAM. With the 15 GB text encoder and the 20 GB H3 Image DiT, which do
not fit together in 32 GB, every prompt change swaps ~8 GB each way and the pinning costs more than it saves.

"Load VAE (BF16)": a VAE loader that runs the VAE with bf16 activations instead of fp16. The MiniMax H3 int8 video
VAE then takes comfy-kitchen's tiled int8 GEMM (as the DiT does) instead of the fp16 WMMA kernel. With fast_rms_rope
on, the attention layers of that VAE (and no other model) use rmsrope.py's Triton RMSNorm+RoPE for head dim 64 instead
of comfy-kitchen's generic HIP kernel (~9x faster per call, closer to an fp32 reference). With fast_convrot_quant
on, int8 GEMM inputs quantized while that VAE decodes go through convrot.py's Triton ConvRot quantizer (~3x faster).
"""
import copy
import threading
import types

import torch
import comfy.model_patcher
import comfy.sd
import comfy.text_encoders.qwen_vl
from comfy.text_encoders import gemma4, llama, minimax, qwen35, qwen3vl

OPTION = "h3accel_te_bf16"
_state = threading.local()

# Both hooks below are inert unless the CLIP being encoded carries OPTION in its patcher's model_options.
_encode_from_tokens = comfy.sd.CLIP.encode_from_tokens


def _encode(self, *args, **kwargs):
    previous = getattr(_state, "bf16", False)
    _state.bf16 = bool(self.patcher.model_options.get(OPTION, False))
    try:
        return _encode_from_tokens(self, *args, **kwargs)
    finally:
        _state.bf16 = previous


_forward = minimax.MiniMaxQwen3VL.forward


def _lm_forward(self, input_ids, attention_mask=None, embeds=None, *args, dtype=None, **kwargs):
    # exact class only: the H3 Image encoder (PH3ImageQwen3VL) subclasses this one
    if getattr(_state, "bf16", False) and type(self) is minimax.MiniMaxQwen3VL and dtype == torch.float32:
        dtype = torch.bfloat16
        if embeds is not None:
            embeds = embeds.to(torch.bfloat16)
    return _forward(self, input_ids, attention_mask, embeds, *args, dtype=dtype, **kwargs)


_img_forward_orig = minimax.PH3ImageQwen3VL.forward


def _img_forward(self, input_ids, attention_mask=None, embeds=None, *args, dtype=None, **kwargs):
    if getattr(_state, "bf16", False) and type(self) is minimax.PH3ImageQwen3VL and dtype == torch.float32:
        if embeds is not None:
            embeds = embeds.to(torch.bfloat16)
        x, hidden_states = _forward(self, input_ids, attention_mask, embeds, *args, dtype=torch.bfloat16, **kwargs)
        return x.float(), self.text_connector(hidden_states.float())
    return _img_forward_orig(self, input_ids, attention_mask, embeds, *args, dtype=dtype, **kwargs)


_generate = comfy.sd.CLIP.generate


def _gen(self, *args, **kwargs):
    # TextGenerate (captioning, prompt enhancers): the image goes through the same fp32 vision tower
    previous = getattr(_state, "bf16", False)
    _state.bf16 = bool(self.patcher.model_options.get(OPTION, False))
    try:
        return _generate(self, *args, **kwargs)
    finally:
        _state.bf16 = previous


comfy.sd.CLIP.encode_from_tokens = _encode
comfy.sd.CLIP.generate = _gen
minimax.MiniMaxQwen3VL.forward = _lm_forward
minimax.PH3ImageQwen3VL.forward = _img_forward

_qwen3vl_forward = qwen3vl.Qwen3VL.forward
_qwen3vl_preprocess = qwen3vl.Qwen3VL.preprocess_embed


def _plain_qwen3vl(self):
    # Qwen-Image 2.1's encoder class is a dynamic subclass of Qwen3VL; the MiniMax ones keep their own handling above
    return getattr(_state, "bf16", False) and not isinstance(self, minimax.MiniMaxQwen3VL)


def _qvl_forward(self, input_ids, attention_mask=None, embeds=None, *args, dtype=None, **kwargs):
    if _plain_qwen3vl(self) and dtype == torch.float32:
        if embeds is not None:
            embeds = embeds.to(torch.bfloat16)
        out = _qwen3vl_forward(self, input_ids, attention_mask, embeds, *args, dtype=torch.bfloat16, **kwargs)
        if isinstance(out, tuple):
            return tuple(o.float() if torch.is_tensor(o) and o.is_floating_point() else o for o in out)
        return out.float() if torch.is_tensor(out) else out
    return _qwen3vl_forward(self, input_ids, attention_mask, embeds, *args, dtype=dtype, **kwargs)


def _qvl_preprocess(self, embed, device):
    if _plain_qwen3vl(self) and embed["type"] == "image":
        image, grid = comfy.text_encoders.qwen_vl.process_qwen2vl_images(
            embed["data"], patch_size=16, image_mean=[0.5, 0.5, 0.5], image_std=[0.5, 0.5, 0.5])
        merged, deepstack = self.visual(image.to(device, dtype=torch.bfloat16), grid)
        return merged, {"grid": grid, "deepstack": deepstack}
    return _qwen3vl_preprocess(self, embed, device)


qwen3vl.Qwen3VL.forward = _qvl_forward
qwen3vl.Qwen3VL.preprocess_embed = _qvl_preprocess

# Qwen3.5 (MOSS Image's qwen35_9b): same treatment, bf16 language model and vision tower, fp32 outputs.
_qwen35_forward = qwen35.Qwen35.forward
_qwen35_preprocess = qwen35.Qwen35.preprocess_embed


def _q35_forward(self, x, attention_mask=None, embeds=None, *args, dtype=None, **kwargs):
    if getattr(_state, "bf16", False) and dtype == torch.float32:
        if embeds is not None:
            embeds = embeds.to(torch.bfloat16)
        out = _qwen35_forward(self, x, attention_mask, embeds, *args, dtype=torch.bfloat16, **kwargs)
        if isinstance(out, tuple):
            return tuple(o.float() if torch.is_tensor(o) and o.is_floating_point() else o for o in out)
        return out.float() if torch.is_tensor(out) else out
    return _qwen35_forward(self, x, attention_mask, embeds, *args, dtype=dtype, **kwargs)


def _q35_preprocess(self, embed, device):
    if getattr(_state, "bf16", False) and embed["type"] == "image":
        image, grid = comfy.text_encoders.qwen_vl.process_qwen2vl_images(
            embed["data"], patch_size=16, image_mean=[0.5, 0.5, 0.5], image_std=[0.5, 0.5, 0.5])
        return self.visual(image.to(device, dtype=torch.bfloat16), grid), grid
    return _qwen35_preprocess(self, embed, device)


qwen35.Qwen35.forward = _q35_forward
qwen35.Qwen35.preprocess_embed = _q35_preprocess

# Qwen3 4B (Z-Image / Lumina 2): text only.
_qwen3_4b_forward = llama.Qwen3_4B.forward


def _q4b_forward(self, input_ids, *args, **kwargs):
    if getattr(_state, "bf16", False) and kwargs.get("dtype") == torch.float32:
        kwargs["dtype"] = torch.bfloat16
        if kwargs.get("embeds") is not None:
            kwargs["embeds"] = kwargs["embeds"].to(torch.bfloat16)
        out = _qwen3_4b_forward(self, input_ids, *args, **kwargs)
        if isinstance(out, tuple):
            return tuple(o.float() if torch.is_tensor(o) and o.is_floating_point() else o for o in out)
        return out.float() if torch.is_tensor(out) else out
    return _qwen3_4b_forward(self, input_ids, *args, **kwargs)


llama.Qwen3_4B.forward = _q4b_forward

# Gemma 4 (LTX-2.5's gemma4-12b text encoder, prompts padded to 1024 tokens): text only. Gemma4Base inherits
# BaseLlama.forward, so the wrapper goes on Gemma4Base and leaves the other BaseLlama models alone.
_gemma4_forward = gemma4.Gemma4Base.forward


def _g4_forward(self, input_ids, *args, **kwargs):
    if getattr(_state, "bf16", False) and kwargs.get("dtype") == torch.float32:
        kwargs["dtype"] = torch.bfloat16
        if kwargs.get("embeds") is not None:
            kwargs["embeds"] = kwargs["embeds"].to(torch.bfloat16)
        out = _gemma4_forward(self, input_ids, *args, **kwargs)
        if isinstance(out, tuple):
            return tuple(o.float() if torch.is_tensor(o) and o.is_floating_point() else o for o in out)
        return out.float() if torch.is_tensor(out) else out
    return _gemma4_forward(self, input_ids, *args, **kwargs)


gemma4.Gemma4Base.forward = _g4_forward

NO_PIN = "h3accel_no_pin"
_pin_weight_to_device = comfy.model_patcher.ModelPatcher.pin_weight_to_device


def _pin(self, key):
    # inert unless this patcher (or the patcher it was cloned from) went through the Unpinned Offload node
    if self.model_options.get(NO_PIN, False):
        return
    return _pin_weight_to_device(self, key)


comfy.model_patcher.ModelPatcher.pin_weight_to_device = _pin


class MiniMaxH3TextEncoderBF16:
    DESCRIPTION = ("Runs Qwen3-VL / Qwen3.5 text encoders (MiniMax H3 video and image, Qwen-Image 2.1, Krea 2, "
                   "MOSS Image) in bf16 instead of fp32: "
                   "same weights file, much faster encode on AMD. Has no effect on other text encoders.")
    CATEGORY = "advanced/model"
    RETURN_TYPES = ("CLIP",)
    FUNCTION = "apply"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"clip": ("CLIP",), "enabled": ("BOOLEAN", {"default": True})}}

    def apply(self, clip, enabled):
        out = clip.clone()
        out.patcher.model_options[OPTION] = enabled
        return (out,)


class MiniMaxH3VAELoaderBF16:
    DESCRIPTION = ("Loads a VAE with bf16 activations instead of the default fp16. For the MiniMax H3 int8 video VAE this "
                   "puts its int8 GEMMs on comfy-kitchen's tiled kernel (the one the DiT uses), 8-17 % faster per call. "
                   "fast_rms_rope: Triton RMSNorm+RoPE for the H3 VAE's attention (head dim 64). fast_convrot_quant: Triton "
                   "ConvRot int8 quantizer for this VAE's decode.")
    CATEGORY = "loaders"
    RETURN_TYPES = ("VAE",)
    FUNCTION = "load"

    @classmethod
    def INPUT_TYPES(cls):
        import folder_paths
        return {"required": {"vae_name": (folder_paths.get_filename_list("vae"),),
                             "fast_rms_rope": ("BOOLEAN", {"default": True}),
                             "fast_convrot_quant": ("BOOLEAN", {"default": True})}}

    def load(self, vae_name, fast_rms_rope=True, fast_convrot_quant=True):
        import comfy.utils
        import folder_paths
        sd, metadata = comfy.utils.load_torch_file(folder_paths.get_full_path_or_raise("vae", vae_name),
                                                   return_metadata=True)
        vae = comfy.sd.VAE(sd=sd, metadata=metadata, dtype=torch.bfloat16)
        vae.throw_exception_if_invalid()
        if fast_rms_rope:
            _use_fast_rms_rope(vae.first_stage_model)
        if fast_convrot_quant:
            _use_fast_convrot(vae)
        return (vae,)


_hip_rotate_quant = None


def _rotate_quant(x2d, group_size, input_act=None, act_weight=None, act_eps=0.0, *, tiled_output=False):
    from . import convrot
    if getattr(_state, "convrot", False) and convrot.supported(x2d, group_size, input_act, act_weight):
        return convrot.rotate_quant_int8(x2d, group_size, input_act, act_weight, act_eps, tiled_output=tiled_output)
    return _hip_rotate_quant(x2d, group_size, input_act, act_weight, act_eps, tiled_output=tiled_output)


def _use_fast_convrot(vae):
    """Route comfy-kitchen's hip ConvRot quantizer to the Triton one, only inside this VAE's decode."""
    global _hip_rotate_quant
    try:
        from comfy_kitchen.backends import hip
    except ImportError:
        return
    if not getattr(hip, "_EXT_AVAILABLE", False) or not hasattr(hip, "_rotate_quant_int8"):
        return
    if _hip_rotate_quant is None:
        _hip_rotate_quant = hip._rotate_quant_int8
        hip._rotate_quant_int8 = _rotate_quant
    decode = vae.decode

    def decode_fast(*args, **kwargs):
        previous = getattr(_state, "convrot", False)
        _state.convrot = True
        try:
            return decode(*args, **kwargs)
        finally:
            _state.convrot = previous

    vae.decode = decode_fast


def _use_fast_rms_rope(model):
    from comfy.ldm.minimax import vae as h3vae
    from . import rmsrope
    _rms_rope = h3vae.Attention.rms_rope

    def rms_rope(self, query, key, rotary_pos_emb):
        if torch.is_grad_enabled():
            return _rms_rope(self, query, key, rotary_pos_emb)
        return rmsrope.rms_rope_split_half_(query, key, rotary_pos_emb, self.qk_norm_scale.to(query.device),
                                            epsilon=self.norm_q.eps, rot_dim=rotary_pos_emb.shape[-3] * 2)

    for m in model.modules():
        if isinstance(m, h3vae.Attention):
            m.rms_rope = types.MethodType(rms_rope, m)


class UnpinnedOffload:
    DESCRIPTION = ("Weights of the MODEL / CLIP passing through are offloaded to RAM without page-locking. Faster model "
                   "swaps when two models do not fit in VRAM together (H3 Image: text encoder <-> DiT on every prompt "
                   "change). Leave the video DiT out: its partial offload during sampling streams from pinned RAM.")
    CATEGORY = "advanced/model"
    RETURN_TYPES = ("MODEL", "CLIP")
    FUNCTION = "apply"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"enabled": ("BOOLEAN", {"default": True})},
                "optional": {"model": ("MODEL",), "clip": ("CLIP",)}}

    def apply(self, enabled, model=None, clip=None):
        if model is not None:
            model = model.clone()
            model.model_options[NO_PIN] = enabled
        if clip is not None:
            clip = clip.clone()
            clip.patcher.model_options[NO_PIN] = enabled
        return (model, clip)


def _install_chunked_convs(model):
    """Give every 3x3x3 conv inside a Lightricks CausalConv3d a forward that can run in temporal slices. Inert unless
    _state.conv_chunk is set (only inside the decode of a VAE that went through the Chunked Conv3D node)."""
    from comfy.ldm.lightricks.vae.causal_conv3d import CausalConv3d
    count = 0
    for m in model.modules():
        if not isinstance(m, CausalConv3d):
            continue
        conv = m.conv
        kt = conv.kernel_size[0]
        if conv.stride[0] != 1 or conv.dilation[0] != 1 or conv.padding[0] != 0 or kt == 1:
            continue
        count += 1
        if getattr(conv, "_h3accel_chunked", False):
            continue
        orig = conv.forward

        def forward(x, orig=orig, kt=kt):
            n = getattr(_state, "conv_chunk", 0)
            t_out = x.shape[2] - kt + 1
            if not n or t_out <= n:
                return orig(x)
            # output frames [i, i + n) only read input frames [i, i + n + kt - 1): same values, smaller workspace
            return torch.cat([orig(x[:, :, i:i + n + kt - 1]) for i in range(0, t_out, n)], dim=2)

        conv.forward = forward
        conv._h3accel_chunked = True
    return count


class VAEChunkedConv3D:
    DESCRIPTION = ("For LTX conv video VAEs (ltx-2.5-video-vae-conv). Without MIOpen, PyTorch runs each 3D convolution "
                   "through a column buffer of 4-5 GB; the buffers of different sizes pile up in PyTorch's cache until "
                   "the card is full, which can hang the AMD driver. This node runs each convolution a few frames at a "
                   "time: same values, about a third of the VRAM, no slower. memory_factor scales ComfyUI's estimate of "
                   "the VRAM the decode needs, which decides how much of the diffusion model is moved out first.")
    CATEGORY = "advanced/model"
    RETURN_TYPES = ("VAE",)
    FUNCTION = "apply"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"vae": ("VAE",),
                             "frames_per_chunk": ("INT", {"default": 4, "min": 0, "max": 64,
                                                          "tooltip": "Output frames per convolution call. 0 = off."}),
                             "memory_factor": ("FLOAT", {"default": 1.5, "min": 0.5, "max": 10.0, "step": 0.1})}}

    def apply(self, vae, frames_per_chunk, memory_factor):
        # shallow copy: the loader's VAE keeps its own estimate and methods; the model and patcher are shared
        out = copy.copy(vae)
        base = vae.memory_used_decode
        out.memory_used_decode = lambda shape, dtype: base(shape, dtype) * memory_factor
        if frames_per_chunk > 0 and _install_chunked_convs(vae.first_stage_model):
            for name in ("decode", "decode_tiled"):
                unbound = getattr(type(vae), name)

                def run(*args, _unbound=unbound, **kwargs):
                    previous = getattr(_state, "conv_chunk", 0)
                    _state.conv_chunk = frames_per_chunk
                    try:
                        return _unbound(out, *args, **kwargs)
                    finally:
                        _state.conv_chunk = previous

                setattr(out, name, run)
        return (out,)


NODE_CLASS_MAPPINGS = {"MiniMaxH3TextEncoderBF16": MiniMaxH3TextEncoderBF16,
                       "MiniMaxH3VAELoaderBF16": MiniMaxH3VAELoaderBF16,
                       "H3UnpinnedOffload": UnpinnedOffload,
                       "H3VAEChunkedConv3D": VAEChunkedConv3D}
NODE_DISPLAY_NAME_MAPPINGS = {"MiniMaxH3TextEncoderBF16": "Text Encoder BF16",
                              "MiniMaxH3VAELoaderBF16": "Load VAE (BF16)",
                              "H3UnpinnedOffload": "Unpinned Offload (fast model swap)",
                              "H3VAEChunkedConv3D": "VAE Chunked Conv3D (LTX conv VAE)"}
