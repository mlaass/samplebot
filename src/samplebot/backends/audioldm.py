"""AudioLDM (v1, small) via diffusers. Ungated, ~1.7 GB, 16 kHz mono, fast. Supports negative prompts."""

import functools
import os

import numpy as np

from samplebot.fetch import model_path

MODEL = os.environ.get("SAMPLEBOT_AUDIOLDM") or model_path("cvssp/audioldm-s-full-v2")
SR = 16000


@functools.cache
def pipe():
    import torch
    from diffusers import AudioLDMPipeline

    p = AudioLDMPipeline.from_pretrained(MODEL, torch_dtype=torch.float16)
    return p.to("cuda" if torch.cuda.is_available() else "cpu")


def generate(prompt: str, negative: str, seconds: float, seed: int, steps: int):
    import torch

    p = pipe()
    out = p(
        prompt,
        negative_prompt=negative or None,
        audio_length_in_s=seconds,
        num_inference_steps=steps or 50,
        generator=torch.Generator(p.device).manual_seed(seed),
    )
    return np.asarray(out.audios[0], dtype=np.float32)[None, :], SR


def unload():
    import gc

    import torch

    pipe.cache_clear()
    gc.collect()
    torch.cuda.empty_cache()
