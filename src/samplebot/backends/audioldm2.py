"""AudioLDM2 via diffusers. Ungated, ~4.5 GB download, 16 kHz mono, supports negative prompts."""

import functools
import os

import numpy as np

MODEL = os.environ.get("SAMPLEBOT_AUDIOLDM2", "cvssp/audioldm2")
SR = 16000


@functools.cache
def pipe():
    import torch
    from diffusers import AudioLDM2Pipeline

    p = AudioLDM2Pipeline.from_pretrained(MODEL, torch_dtype=torch.float16)
    return p.to("cuda" if torch.cuda.is_available() else "cpu")


def generate(prompt: str, negative: str, seconds: float, seed: int, steps: int):
    import torch

    p = pipe()
    out = p(
        prompt,
        negative_prompt=negative or None,
        audio_length_in_s=seconds,
        num_inference_steps=steps or 200,
        generator=torch.Generator(p.device).manual_seed(seed),
    )
    return np.asarray(out.audios[0], dtype=np.float32)[None, :], SR
