"""Deterministic test backend: a decaying tone whose pitch depends on the prompt. No GPU, no download."""

import zlib

import numpy as np

SR = 16000


def generate(prompt: str, negative: str, seconds: float, seed: int, steps: int):
    freq = 200 + zlib.crc32(prompt.encode()) % 800
    t = np.arange(int(seconds * SR)) / SR
    rng = np.random.default_rng(seed)
    tone = np.sin(2 * np.pi * freq * t) * np.exp(-t) + 0.01 * rng.standard_normal(t.size)
    return tone.astype(np.float32)[None, :] * 0.5, SR
