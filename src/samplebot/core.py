import importlib
import json
import os
import re
import subprocess
import time
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


_loaded = None  # ponytail: one model resident at a time, the GPU has 16 GB and moss alone takes 9


def generate(prompt: Prompt, model: str = "fake", seconds: float = 5.0, seed: int = 0, steps: int = 0):
    """Returns (audio float32 array of shape [channels, samples], sample_rate)."""
    global _loaded
    if model not in BACKENDS:
        raise ValueError(f"unknown model {model!r}, choose from {', '.join(BACKENDS)}")
    backend = importlib.import_module(BACKENDS[model])
    if _loaded is not None and _loaded is not backend:
        _loaded.unload()  # free the previous model's VRAM before loading another
    _loaded = backend
    audio, sr = backend.generate(prompt.positive_text(), prompt.negative_text(), seconds, seed, steps)
    audio = np.atleast_2d(np.asarray(audio, dtype=np.float32))
    return audio, sr


def unload() -> str | None:
    """Free the resident model. Returns its name, or None if nothing was loaded."""
    global _loaded
    if _loaded is None:
        return None
    _loaded.unload()
    name = next(k for k, v in BACKENDS.items() if v == _loaded.__name__)
    _loaded = None
    return name


def write_wav(path: Path, audio: np.ndarray, sr: int, meta: dict | None = None) -> Path:
    """16-bit PCM WAV via stdlib. audio is [channels, samples] in [-1, 1]. Writes a .json sidecar if meta given."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    audio = audio / max(1.0, float(np.abs(audio).max()))  # scale down only if a backend overshoots full scale (stable-audio does)
    pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2").T  # [T, C] interleaved
    tmp = path.with_suffix(".tmp.wav")  # renamed last, so an existing .wav is always complete (batch resume relies on it)
    with wave.open(str(tmp), "wb") as w:
        w.setnchannels(audio.shape[0])
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    if meta is not None:
        path.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    os.replace(tmp, path)
    return path


def resample(audio: np.ndarray, sr: int, rate: int) -> np.ndarray:
    """[C, T] float32 from sr to rate through ffmpeg's resampler, piped as raw float32. No Python dependency."""
    if sr == rate:
        return audio
    channels = audio.shape[0]
    cmd = ["ffmpeg", "-v", "error", "-f", "f32le", "-ar", str(sr), "-ac", str(channels), "-i", "pipe:0", "-ar", str(rate), "-f", "f32le", "pipe:1"]
    try:
        r = subprocess.run(cmd, input=audio.T.astype("<f4").tobytes(), capture_output=True, check=True)
    except FileNotFoundError:
        raise RuntimeError("resampling needs ffmpeg on PATH") from None
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffmpeg failed: {e.stderr.decode().strip()}") from None
    return np.frombuffer(r.stdout, "<f4").reshape(-1, channels).T


def prompt_meta(prompt: Prompt, **kw) -> dict:
    return {**asdict(prompt), **kw}


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:60] or "sound"


def run(prompt: Prompt, model: str = "fake", seconds: float = 5.0, seed: int = 0, steps: int = 0, count: int = 1,
        out_dir: Path = Path("out"), out: Path | None = None, optimize: str | None = None, rate: int | None = None,
        extra: dict | None = None) -> list[Path]:
    """Generate `count` variations (seed increments) and write wav + json sidecar. optimize = Ollama model name or "" for default.
    rate resamples the output with ffmpeg; extra is merged into the sidecar."""
    source = prompt.text
    if optimize is not None:
        from samplebot.optimize import DEFAULT_LLM, optimize as _optimize

        # keywords go into the LLM too, so they are baked into the rewritten text and not appended twice
        prompt = Prompt(_optimize(prompt.positive_text(), optimize or DEFAULT_LLM), negative=prompt.negative)
    paths = []
    for i in range(count):
        s = seed + i
        audio, sr = generate(prompt, model, seconds, s, steps)
        if rate:
            audio = resample(audio, sr, rate)
        path = out or Path(out_dir) / f"{slug(prompt.text)}-{model}-{s}.wav"
        if out and count > 1:
            path = out.with_stem(f"{out.stem}-{s}")
        meta = prompt_meta(prompt, source=source, model=model, seconds=seconds, seed=s, steps=steps, sample_rate=rate or sr, native_rate=sr,
                           **(extra or {}))
        paths.append(write_wav(path, audio, rate or sr, meta))
    return paths


def batch(jobs: list[dict], rate: int | None = None, force: bool = False) -> int:
    """Run generation jobs ({id, text, positive, negative, model, seconds, seed, steps, out}) grouped by model, so each model
    loads once. Jobs whose out already exists are skipped unless force. A failing job is reported and the rest still run.
    Returns the number of failed jobs."""
    order = {}
    for j in jobs:
        order.setdefault(j.get("model"), len(order))
    todo = [j for j in jobs if force or not ("out" in j and Path(j["out"]).exists())]
    todo.sort(key=lambda j: order[j.get("model")])  # stable: file order is kept within a model
    per_model = ", ".join(f"{m} {n}" for m in order if (n := sum(j.get("model") == m for j in todo)))
    print(f"batch: {len(todo)} jobs ({per_model or 'none'}), {len(jobs) - len(todo)} already done", flush=True)
    failed = 0
    for i, j in enumerate(todo, 1):
        head = f"[{i}/{len(todo)}] {j.get('model')} {j.get('id', j.get('out'))}"
        start = time.monotonic()
        try:
            prompt = Prompt(j["text"], list(j.get("positive", [])), list(j.get("negative", [])))
            seconds = float(j.get("seconds", 5.0))
            run(prompt, j["model"], seconds, int(j.get("seed", 0)), int(j.get("steps", 0)), out=Path(j["out"]), rate=rate,
                extra={"id": j["id"]} if "id" in j else None)
            print(f"{head}  {seconds:g}s  took {time.monotonic() - start:.1f}s", flush=True)
        except Exception as e:  # one bad job (CUDA OOM, typo in a model name) must not stop an overnight run
            failed += 1
            print(f"{head}  FAILED: {type(e).__name__}: {e}", flush=True)
    print(f"batch: {len(todo) - failed} generated, {failed} failed", flush=True)
    return failed
