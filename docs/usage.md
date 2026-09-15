# Usage

## Install

```bash
uv sync                # CLI, fake backend, tests (no GPU needed)
uv sync --extra gpu    # + torch/diffusers for audioldm2 and stable-audio (~4 GB)
```

## Generate

```bash
uv run samplebot gen "heavy rain hitting a tin roof" -m audioldm2 -s 5
uv run samplebot gen "footsteps on gravel" -p close-up -p crunchy -n music -n speech -m audioldm2 --seed 3 -c 4
uv run samplebot gen "door creak" -m audioldm2 --optimize            # rewrite prompt with qwen2.5:7b-instruct first
uv run samplebot gen "door creak" -m audioldm2 --optimize qwen3:14b  # pick the Ollama model
uv run samplebot gen "wind" -m fake -o /tmp/wind.wav                 # explicit output path
```

Flags:

| Flag | Meaning |
|---|---|
| `-p/--positive KW` | append a positive keyword (repeatable) |
| `-n/--negative KW` | negative keyword (repeatable); passed as `negative_prompt` to models that support it |
| `-m/--model` | `fake`, `audioldm2`, `stable-audio` (`samplebot models` lists them) |
| `-s/--seconds` | length in seconds |
| `--seed`, `-c/--count` | seed and number of variations (seed increments) |
| `--steps` | inference steps, `0` = backend default (audioldm2: 200, stable-audio: 100) |
| `--optimize [LLM]` | rewrite the prompt with a local Ollama model first |
| `-o/--out` | output path; default `out/<slug>-<model>-<seed>.wav` |

Every WAV gets a `.json` sidecar with all parameters.

## Optimize only

```bash
uv run samplebot optimize "rain on roof"
# → raindrops hitting a metal roof from close-up perspective
```

Env: `OLLAMA_HOST` (default `http://localhost:11434`), `SAMPLEBOT_LLM` (default `qwen2.5:7b-instruct`).

## Dashboard

```bash
uv run samplebot serve            # http://127.0.0.1:7333/ , serves ./out (falls back to a free port if 7333 is taken)
uv run samplebot serve -d ~/sfx --port 9000
```

Lists every run (newest first) with its prompt, keywords, model, seconds, seed and steps, with an inline player and a
text filter, plus a "play all" button and an "autoplay next" toggle that plays the listed runs in order.

The form at the top generates from the browser with the same controls as the CLI (prompt, positive/negative keywords,
model, seconds, seed, steps, count, optional LLM optimization). Each run has two buttons:

- **reuse** loads that run's prompt and parameters into the form so you can tweak and regenerate.
- **variation** regenerates immediately with the same parameters and the next seed.

Generation runs inside the server process, one at a time, so the page waits while the GPU works (about 10 s for a
5 s AudioLDM2 clip). Stdlib `http.server`, no build step: it reads the `.json` sidecars next to the `.wav` files.

HTTP API (what the page uses):

```
GET  /api/models                 -> ["fake", "audioldm2", "stable-audio"]
GET  /api/runs                   -> [{text, positive, negative, model, seconds, seed, steps, wav, mtime, ...}]
POST /api/generate  {"text": "...", "negative": ["music"], "model": "audioldm2", "seconds": 5, "seed": 0, "steps": 0, "count": 1, "optimize": null}
                                 -> {"paths": ["....wav"]}  or 500 {"error": "..."}
```

## Python API

```python
from samplebot import Prompt, generate, write_wav

audio, sr = generate(Prompt("dog barking", negative=["music"]), model="audioldm2", seconds=5, seed=1)
write_wav("dog.wav", audio, sr)
```

## Downloads

Models are pulled from Hugging Face into `~/.cache/huggingface` on first use (AudioLDM2 ≈ 4.5 GB). If the direct
CDN is slow (it was ~150 KB/s here), use the mirror:

```bash
export HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1
```

After the first download, `HF_HUB_OFFLINE=1` skips the Hub round-trip and loads straight from cache.

## Models

- **audioldm2** — `cvssp/audioldm2`, ungated, 16 kHz mono. ~7 s per 5 s clip at 50 steps on an RTX 5060 Ti (fp16). Needs `transformers<4.50` (pinned): newer versions drop a GPT2 helper the diffusers pipeline still calls. Override with `SAMPLEBOT_AUDIOLDM2=cvssp/audioldm2-large`.
- **stable-audio** — `stabilityai/stable-audio-open-1.0`, 44.1 kHz stereo, up to 47 s. Gated: accept the license on
  Hugging Face, then `export HF_TOKEN=hf_...` (or `uv run hf auth login`).

## Tests

```bash
uv run pytest
```
