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
| Hugging Face token | none. Stable Audio Open is gated → blocked until a token with accepted license is set |

## Model choice (from docs/start.md)

| Backend | Status | Why |
|---|---|---|
| `fake` | done | deterministic tone; makes tests and the CLI runnable without GPU |
| `audioldm2` | done | ungated, in `diffusers`, native `negative_prompt`, 16 kHz mono, ≤ ~10 s useful |
| `stable-audio` | code done, untested | gated repo; needs `HF_TOKEN`. 44.1 kHz stereo, ≤ 47 s, native `negative_prompt` |
| MOSS-SoundEffect v2 | deferred | 11 GB, own package (`moss_soundeffect_v2`, torch.compile+Triton), no negative prompt. Best quality candidate once disk is freed |
| AudioGen | deferred | `audiocraft` pins old torch; conflicts with sm_120 wheels. Run in Docker if ever needed |
| Tango 2 | deferred | custom repo code, no negative prompt, superseded by the above |

## Phases

1. Package skeleton (uv), `Prompt` model, `fake` backend, argparse CLI, Ollama optimizer, tests. ✅
2. GPU backends: AudioLDM2 (verified: rain, footsteps, 7 s per clip), Stable Audio Open (code only, needs token). Pinned `transformers<4.50`. ✅
3. Docs: this plan, usage. ✅
3b. Dashboard: `samplebot serve`, stdlib http.server + one HTML page over the `.json` sidecars. ✅
4. Next iteration candidates (not built, add when needed):
   - HTTP API (FastAPI wrapper around `samplebot.generate`) once something other than the shell calls it.
   - MOSS-SoundEffect v2 backend behind an extra, once ~12 GB disk is available.
   - Batch mode from a JSON/CSV list of prompts.
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
