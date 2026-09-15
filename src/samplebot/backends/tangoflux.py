"""TangoFlux (declare-lab) — 44.1 kHz stereo, up to 30 s, rectified flow, CC-BY-NC. Ungated, ~4 GB + flan-t5-large (~3 GB).

Inference vendored from the model repo's model.py (inference_flow) so we do not pull the upstream package,
which pins torch 2.4 / transformers 4.44. The "negative" text replaces the empty unconditional prompt in CFG.
"""

import functools
import json
import os
from math import pi

import numpy as np

from samplebot.fetch import model_path

MODEL = os.environ.get("SAMPLEBOT_TANGOFLUX") or model_path("declare-lab/TangoFlux")
SR = 44100


@functools.cache
def load():
    import torch
    from diffusers import AutoencoderOobleck, FlowMatchEulerDiscreteScheduler, FluxTransformer2DModel
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file
    from torch import nn
    from transformers import T5EncoderModel, T5TokenizerFast

    class PositionalEmbedding(nn.Module):  # StableAudioPositionalEmbedding
        def __init__(self, dim):
            super().__init__()
            self.weights = nn.Parameter(torch.randn(dim // 2))

        def forward(self, t):
            f = t[..., None] * self.weights[None] * 2 * pi
            return torch.cat((t[..., None], f.sin(), f.cos()), dim=-1)

    class DurationEmbedder(nn.Module):
        def __init__(self, dim, max_value, internal=256):
            super().__init__()
            self.time_positional_embedding = nn.Sequential(PositionalEmbedding(internal), nn.Linear(internal + 1, dim))
            self.max_value, self.dim = max_value, dim

        def forward(self, seconds):
            x = seconds.clamp(0, self.max_value) / self.max_value
            return self.time_positional_embedding(x.to(next(self.parameters()).dtype)).view(-1, 1, self.dim)

    class TangoFlux(nn.Module):  # attribute names match the checkpoint keys
        def __init__(self, cfg):
            super().__init__()
            self.cfg = cfg
            self.text_encoder = T5EncoderModel.from_pretrained(model_path(cfg["text_encoder_name"]))
            self.tokenizer = T5TokenizerFast.from_pretrained(model_path(cfg["text_encoder_name"]))
            d = self.text_encoder.config.d_model
            self.fc = nn.Sequential(nn.Linear(d, cfg["joint_attention_dim"]), nn.ReLU())
            self.duration_emebdder = DurationEmbedder(d, cfg["max_duration"])
            self.transformer = FluxTransformer2DModel(
                in_channels=cfg["in_channels"], num_layers=cfg["num_layers"], num_single_layers=cfg["num_single_layers"],
                attention_head_dim=cfg["attention_head_dim"], num_attention_heads=cfg["num_attention_heads"],
                joint_attention_dim=cfg["joint_attention_dim"], pooled_projection_dim=d, guidance_embeds=False)

        @torch.no_grad()
        def encode(self, texts, max_length=None):
            dev = self.text_encoder.device
            b = self.tokenizer(texts, max_length=max_length or self.tokenizer.model_max_length,
                               padding="max_length" if max_length else True, truncation=True, return_tensors="pt")
            return self.text_encoder(input_ids=b.input_ids.to(dev), attention_mask=b.attention_mask.to(dev))[0], b.attention_mask.to(dev) == 1

    path = MODEL if os.path.isdir(MODEL) else snapshot_download(MODEL, allow_patterns=["*.json", "*.safetensors"])
    cfg = json.load(open(f"{path}/config.json"))
    model = TangoFlux(cfg)
    model.load_state_dict(load_file(f"{path}/tangoflux.safetensors"), strict=False)  # text_encoder embed_tokens missing: expected
    vae = AutoencoderOobleck()
    vae.load_state_dict(load_file(f"{path}/vae.safetensors"))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    return model.to(dev).eval(), vae.to(dev).eval(), FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000)


def generate(prompt: str, negative: str, seconds: float, seed: int, steps: int, guidance: float = 4.5):
    import torch

    model, vae, sched = load()
    dev, cfg = model.transformer.device, model.cfg
    steps = steps or 50
    seconds = min(seconds, cfg["max_duration"])
    with torch.no_grad():
        emb, mask = model.encode([prompt])
        uncond, umask = model.encode([negative or ""], max_length=emb.shape[1])
        emb, mask = torch.cat([uncond, emb]), torch.cat([umask, mask])
        pooled = model.fc(torch.nanmean(torch.where(mask[..., None], emb, torch.nan), dim=1))
        dur = model.duration_emebdder(torch.tensor([float(seconds)], device=dev)).repeat(2, 1, 1)
        emb = torch.cat([emb, dur], dim=1)
        sched.set_timesteps(sigmas=np.linspace(1.0, 1 / steps, steps), device=dev)
        n = cfg["audio_seq_len"]
        latents = torch.randn(1, n, cfg["in_channels"], generator=torch.Generator().manual_seed(seed)).to(dev)
        txt_ids = torch.zeros(emb.shape[1], 3, device=dev)
        audio_ids = torch.arange(n, device=dev)[:, None].repeat(1, 3).float()
        for t in sched.timesteps:
            pred = model.transformer(hidden_states=torch.cat([latents] * 2), timestep=torch.tensor([t / 1000], device=dev),
                                     guidance=None, pooled_projections=pooled, encoder_hidden_states=emb,
                                     txt_ids=txt_ids, img_ids=audio_ids, return_dict=False)[0]
            u, c = pred.chunk(2)
            latents = sched.step(u + guidance * (c - u), t, latents).prev_sample
        wave = vae.decode(latents.transpose(2, 1)).sample[0, :, : int(seconds * SR)]
    return wave.float().cpu().numpy(), SR


def unload():
    import gc

    import torch

    load.cache_clear()
    gc.collect()
    torch.cuda.empty_cache()
