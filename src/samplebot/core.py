import importlib
import json
import re
import wave
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np

# name -> module path. Each module exposes generate(prompt, negative, seconds, seed, steps) -> (audio[C, T] float32, sample_rate)
BACKENDS = {
    "fake": "samplebot.backends.fake",
    "audioldm": "samplebot.backends.audioldm",
    "audioldm2": "samplebot.backends.audioldm2",
    "stable-audio": "samplebot.backends.stable_audio",
    "tangoflux": "samplebot.backends.tangoflux",
    "moss": "samplebot.backends.moss",
}


@dataclass
class Prompt:
    text: str
    positive: list[str] = field(default_factory=list)
    negative: list[str] = field(default_factory=list)

    def positive_text(self) -> str:
        return ", ".join([self.text.strip(), *(k.strip() for k in self.positive)]).strip(", ")

    def negative_text(self) -> str:
        return ", ".join(k.strip() for k in self.negative)


def generate(prompt: Prompt, model: str = "fake", seconds: float = 5.0, seed: int = 0, steps: int = 0):
    """Returns (audio float32 array of shape [channels, samples], sample_rate)."""
    if model not in BACKENDS:
        raise ValueError(f"unknown model {model!r}, choose from {', '.join(BACKENDS)}")
    backend = importlib.import_module(BACKENDS[model])
    audio, sr = backend.generate(prompt.positive_text(), prompt.negative_text(), seconds, seed, steps)
    audio = np.atleast_2d(np.asarray(audio, dtype=np.float32))
    return audio, sr


def write_wav(path: Path, audio: np.ndarray, sr: int, meta: dict | None = None) -> Path:
    """16-bit PCM WAV via stdlib. audio is [channels, samples] in [-1, 1]. Writes a .json sidecar if meta given."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    audio = audio / max(1.0, float(np.abs(audio).max()))  # scale down only if a backend overshoots full scale (stable-audio does)
    pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2").T  # [T, C] interleaved
    with wave.open(str(path), "wb") as w:
        w.setnchannels(audio.shape[0])
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    if meta is not None:
        path.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    return path


def prompt_meta(prompt: Prompt, **kw) -> dict:
    return {**asdict(prompt), **kw}


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:60] or "sound"


def run(prompt: Prompt, model: str = "fake", seconds: float = 5.0, seed: int = 0, steps: int = 0, count: int = 1,
        out_dir: Path = Path("out"), out: Path | None = None, optimize: str | None = None) -> list[Path]:
    """Generate `count` variations (seed increments) and write wav + json sidecar. optimize = Ollama model name or "" for default."""
    source = prompt.text
    if optimize is not None:
        from samplebot.optimize import DEFAULT_LLM, optimize as _optimize

        # keywords go into the LLM too, so they are baked into the rewritten text and not appended twice
        prompt = Prompt(_optimize(prompt.positive_text(), optimize or DEFAULT_LLM), negative=prompt.negative)
    paths = []
    for i in range(count):
        s = seed + i
        audio, sr = generate(prompt, model, seconds, s, steps)
        path = out or Path(out_dir) / f"{slug(prompt.text)}-{model}-{s}.wav"
        if out and count > 1:
            path = out.with_stem(f"{out.stem}-{s}")
        meta = prompt_meta(prompt, source=source, model=model, seconds=seconds, seed=s, steps=steps, sample_rate=sr)
        paths.append(write_wav(path, audio, sr, meta))
    return paths
