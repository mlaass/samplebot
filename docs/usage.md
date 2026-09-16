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
| `--rate HZ` | resample the output to this sample rate with ffmpeg (e.g. `44100`; moss is 48 kHz, audioldm 16 kHz) |

Every WAV gets a `.json` sidecar with all parameters, including `sample_rate` (written) and `native_rate` (the model's).
WAVs are written to `<name>.tmp.wav` and renamed, so an existing `.wav` is always complete.

## Batch

```bash
uv run samplebot batch jobs.json --rate 44100      # or: ... batch - < jobs.json
```

`jobs.json` is a list of jobs. `text`, `model` and `out` are required; the rest default like `gen`:

```json
[{"id": "axe_chop/r1-A-moss-1", "text": "axe chopping into a tree trunk", "positive": [], "negative": ["music"],
  "model": "moss", "seconds": 1.5, "seed": 1, "steps": 0, "out": "/abs/path/axe_chop/r1-A-moss-1.wav"}]
```

- Jobs run grouped by model (in order of first appearance, file order within a model), so each model loads once.
- Jobs whose `out` exists are skipped: rerun after an interruption and it resumes. `--force` regenerates them.
- A failing job (CUDA OOM, unknown model, missing key) is printed with its id and the batch continues; exit code 1 if
  any failed.
- `id` is copied into the sidecar. Use absolute `out` paths: samplebot runs from its own directory (models, `.env`).

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

The dashboard keeps the last model on the GPU between requests. To free the VRAM without stopping the server:

```bash
uv run samplebot unload              # add --port N if serve fell back to another port
```

It waits for a running generation to finish. The next generate reloads the model. `gen` and `batch` free their model on
exit, so a running dashboard is the only thing this needs to reach.

HTTP API (what the page uses):

```
GET  /api/models                 -> ["fake", "audioldm2", "stable-audio"]
GET  /api/runs                   -> [{text, positive, negative, model, seconds, seed, steps, wav, mtime, ...}]
POST /api/generate  {"text": "...", "negative": ["music"], "model": "audioldm2", "seconds": 5, "seed": 0, "steps": 0, "count": 1, "optimize": null,
                     "rate": 44100, "out": "axe_chop/r2-ui1-moss-12.wav"}
                                 -> {"paths": ["....wav"]}  or 400/500 {"error": "..."}
POST /api/unload                 -> {"unloaded": "moss"}  or {"unloaded": null} if nothing was loaded
```

`rate` and `out` are optional. `out` is a `.wav` path relative to the served directory; absolute paths, paths that escape
the directory, and `out` with `count` > 1 are rejected with 400 (as are empty prompts and unknown models). Other
failures are 500.

## Python API

```python
from samplebot import Prompt, generate, write_wav

audio, sr = generate(Prompt("dog barking", negative=["music"]), model="audioldm2", seconds=5, seed=1)
write_wav("dog.wav", audio, sr)
```

## Downloads

Models are pulled from Hugging Face into `~/.cache/huggingface` on first use (sizes in the table below).

The CDN throttles per connection (~250 KB/s here). `hf_xet` (the default) stalls on it and `hf_transfer` gets
rate-limited with its 100 connections and throws the partial file away, so samplebot has its own resumable
range downloader:

```bash
uv run samplebot fetch cvssp/audioldm-s-full-v2 declare-lab/TangoFlux google/flan-t5-large -j 40
```

Files land in `models/<repo>/` (override with `SAMPLEBOT_MODELS`) and the backends use that copy when it exists.
Interrupt and rerun freely: finished chunks are recorded and skipped. 40 workers gave ~3.5 MB/s here.

After the first download, `HF_HUB_OFFLINE=1` skips the Hub round-trip and loads straight from cache.

## Models

| name | model | output | negative prompt | download | notes |
|---|---|---|---|---|---|
| `fake` | none | 16 kHz mono | ignored | none | deterministic tone for tests |
| `audioldm` | `cvssp/audioldm-s-full-v2` | 16 kHz mono | yes | 1.7 GB | fastest: 5 s clip in ~5 s incl. load, default 50 steps |
| `audioldm2` | `cvssp/audioldm2` | 16 kHz mono | yes | 4.5 GB | default 200 steps, ~7 s per 5 s clip at 50 |
| `tangoflux` | `declare-lab/TangoFlux` | 44.1 kHz stereo, ≤ 30 s | yes (used as the CFG unconditional text) | 4 GB + 3 GB flan-t5-large | CC-BY-NC, inference vendored in `backends/tangoflux.py`; 5 s clip in ~14 s at 50 steps |
| `moss` | `OpenMOSS-Team/MOSS-SoundEffect-v2.0` | 48 kHz, ≤ 30 s | yes | 11 GB | Apache-2.0, needs the separate venv below, default 100 steps (~26 s per 5 s clip after the first call); output is quiet, normalize downstream |
| `stable-audio` | `stabilityai/stable-audio-open-1.0` | 44.1 kHz stereo, ≤ 47 s | yes | 5 GB | gated, needs `HF_TOKEN`; 5 s clip in ~19 s at 100 steps |

- **audioldm2** — `cvssp/audioldm2`, ungated, 16 kHz mono. ~7 s per 5 s clip at 50 steps on an RTX 5060 Ti (fp16). Needs `transformers<4.50` (pinned): newer versions drop a GPT2 helper the diffusers pipeline still calls. Override with `SAMPLEBOT_AUDIOLDM2=cvssp/audioldm2-large`.
- **stable-audio** — `stabilityai/stable-audio-open-1.0`, 44.1 kHz stereo, up to 47 s. Gated: accept the license on
  the model page, create a read token under Settings → Access Tokens, and put `HF_TOKEN=hf_...` in `.env` (git-ignored,
  the CLI loads it). Fetch only the diffusers layout:

  ```bash
  uv run samplebot fetch stabilityai/stable-audio-open-1.0 -j 40 -x model.safetensors -x "*.ckpt" -x "*.csv" -x "*.png"
  ```
- **moss** — MOSS-SoundEffect v2 pins `transformers==4.57.1` and `diffusers==0.37.1`, which conflict with the AudioLDM2
  pin, so it gets its own venv and runs as a worker subprocess (`backends/moss_worker.py`). One-time setup:

  ```bash
  uv venv vendor/moss-venv --python 3.12
  uv pip install --python vendor/moss-venv/bin/python torch torchaudio hf_transfer \
      "moss-soundeffect-v2 @ git+https://github.com/OpenMOSS/MOSS-TTS#subdirectory=moss_soundeffect_v2"
  ```

  Override the interpreter with `SAMPLEBOT_MOSS_PYTHON`. The first call compiles the DiT with torch.compile (minutes);
  `TORCHDYNAMO_DISABLE=1` skips that if Triton misbehaves.

## Tests

```bash
uv run pytest
```
