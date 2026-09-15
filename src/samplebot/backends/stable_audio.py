"""Stable Audio Open 1.0 via diffusers. GATED: accept the license on Hugging Face and set HF_TOKEN. 44.1 kHz stereo, up to 47 s."""

import functools
import os

import numpy as np

from samplebot.fetch import model_path

MODEL = os.environ.get("SAMPLEBOT_STABLE_AUDIO") or model_path("stabilityai/stable-audio-open-1.0")


@functools.cache
def pipe():
    import torch
    from diffusers import StableAudioPipeline

    p = StableAudioPipeline.from_pretrained(MODEL, torch_dtype=torch.float16)
    return p.to("cuda" if torch.cuda.is_available() else "cpu")


def generate(prompt: str, negative: str, seconds: float, seed: int, steps: int):
    import torch

    p = pipe()
    out = p(
        prompt,
        negative_prompt=negative or None,
        audio_end_in_s=seconds,
        num_inference_steps=steps or 100,
        num_waveforms_per_prompt=1,
        generator=torch.Generator(p.device).manual_seed(seed),
    )
    audio = out.audios[0].float().cpu().numpy()  # [C, T]
    return np.asarray(audio, dtype=np.float32), p.vae.sampling_rate


def unload():
    import gc

    import torch

    pipe.cache_clear()
    gc.collect()
    torch.cuda.empty_cache()
