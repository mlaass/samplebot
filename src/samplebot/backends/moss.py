"""MOSS-SoundEffect v2 (OpenMOSS) — 48 kHz, up to 30 s, DiT + flow matching, Apache-2.0. ~11 GB download.

Its package pins transformers/diffusers versions that conflict with ours, so it lives in vendor/moss-venv
(see docs/usage.md) and runs as a persistent worker subprocess; the model loads once per samplebot process.
"""

import functools
import json
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
PYTHON = os.environ.get("SAMPLEBOT_MOSS_PYTHON", str(ROOT / "vendor/moss-venv/bin/python"))
WORKER = Path(__file__).with_name("moss_worker.py")


@functools.cache
def worker() -> subprocess.Popen:
    if not Path(PYTHON).exists():
        raise RuntimeError(f"{PYTHON} not found: create the MOSS venv first (docs/usage.md, 'MOSS-SoundEffect')")
    p = subprocess.Popen([PYTHON, "-u", str(WORKER)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
                         env={**os.environ, "HF_HUB_ENABLE_HF_TRANSFER": os.environ.get("HF_HUB_ENABLE_HF_TRANSFER", "1")})
    line = p.stdout.readline()  # blocks until the model is loaded
    if not line:
        raise RuntimeError(f"moss worker exited with {p.wait()} before becoming ready")
    return p


def generate(prompt: str, negative: str, seconds: float, seed: int, steps: int):
    p = worker()
    out = tempfile.NamedTemporaryFile(suffix=".npy", delete=False).name
    try:
        p.stdin.write(json.dumps({"prompt": prompt, "negative": negative, "seconds": seconds, "seed": seed, "steps": steps or 100, "out": out}) + "\n")
        p.stdin.flush()
        line = p.stdout.readline()
        if not line:
            worker.cache_clear()
            raise RuntimeError(f"moss worker died (exit {p.wait()})")
        reply = json.loads(line)
        if "error" in reply:
            raise RuntimeError(reply["error"])
        return np.load(out).astype(np.float32), reply["sr"]
    finally:
        Path(out).unlink(missing_ok=True)
