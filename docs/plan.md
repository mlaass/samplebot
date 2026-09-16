# Plan

Goal: a CLI-first API that turns a text prompt into a sound effect using local open-weight models, with control
over positive/negative keywords and optional prompt optimization by a local LLM (Ollama).

## Constraints found on this machine (2026-09-15)

| Item | State |
|---|---|
| GPU | RTX 5060 Ti, 16 GB VRAM, sm_120 → needs torch ≥ 2.8 (cu128/cu130 wheels) |
| Disk | ~15 GB free before install; torch+CUDA ≈ 4 GB, AudioLDM2 ≈ 4.5 GB |
| Ollama | running on :11434 with qwen2.5:7b-instruct, qwen3:14b, llama3.1:8b, ... |
| HF download speed | ~150 KB/s direct, ~6 MB/s via `HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1` |
| Hugging Face token | in `.env` (git-ignored, loaded by the CLI) since 2026-09-15. Was: none. Stable Audio Open is gated → blocked until a token with accepted license is set |

## Model choice (from docs/start.md)

| Backend | Status | Why |
|---|---|---|
| `fake` | done | deterministic tone; makes tests and the CLI runnable without GPU |
| `audioldm` | done, verified | AudioLDM v1 small, ungated, 1.7 GB, fastest (5 s clip in ~5 s incl. load); same API as audioldm2 |
| `audioldm2` | done | ungated, in `diffusers`, native `negative_prompt`, 16 kHz mono, ≤ ~10 s useful |
| `tangoflux` | done, verified (5 s stereo clip in ~10 s incl. load at 25 steps) | ungated, 44.1 kHz stereo, ≤ 30 s, CC-BY-NC. Upstream package pins torch 2.4, so the ~90-line inference path is vendored |
| `moss` | done, verified (5 s clip, 100 steps ≈ 2 min on first call incl. warm-up, ~26 s after; output level is low, peak ≈ 0.13) | MOSS-SoundEffect v2, 48 kHz, ≤ 30 s, Apache-2.0, 11 GB. Pins conflict with ours → own venv + worker subprocess. Emits recoverable CUDA OOM allocator warnings on 16 GB |
| `stable-audio` | done, verified (5 s stereo clip in ~19 s incl. load at 100 steps) | gated repo; needs `HF_TOKEN` in `.env`. 44.1 kHz stereo, ≤ 47 s, native `negative_prompt` |
| AudioGen | skipped | `audiocraft` pins torch 2.1, which has no sm_120 kernels for this GPU. Superseded by tangoflux/moss |
| Tango 2 | skipped | 16 kHz, custom repo code, no negative prompt; TangoFlux is its successor |

## Phases

1. Package skeleton (uv), `Prompt` model, `fake` backend, argparse CLI, Ollama optimizer, tests. ✅
2. GPU backends: AudioLDM2 (verified: rain, footsteps, 7 s per clip), Stable Audio Open (code only, needs token). Pinned `transformers<4.50`. ✅
3. Docs: this plan, usage. ✅
3b. Dashboard: `samplebot serve`, stdlib http.server + one HTML page over the `.json` sidecars. ✅
3c. Generate from the dashboard (form + `POST /api/generate`), reuse / variation buttons on every run. ✅
4. More models: `audioldm`, `tangoflux`, `moss` backends. ✅ (see table above)
5. `samplebot batch` (JSON jobs grouped by model, resumable, failures don't stop the run), `--rate` resampling via
   ffmpeg, `out`/`rate` on `POST /api/generate`. Built for the settlement SFX pipeline
   (`../settlement/docs/prd-sfx-generation-pipeline.md`, phase 1). ✅
6. Next iteration candidates (not built, add when needed):
   - Optimizer that also proposes negative keywords.

## Architecture (deliberately small)

```
samplebot gen "..." → cli.py → core.generate(Prompt, model) → backends/<model>.generate() → core.write_wav()
                     └── --optimize → optimize.optimize() → Ollama /api/chat
```

- A backend is a module with one function: `generate(prompt, negative, seconds, seed, steps) -> (float32[C, T], sr)`.
  Registered by name in `core.BACKENDS`. No base class.
- Heavy imports (torch, diffusers) live inside backend modules so the CLI and tests start instantly.
- Every WAV gets a `.json` sidecar with prompt, keywords, model, seed, steps, so any sample can be regenerated.
