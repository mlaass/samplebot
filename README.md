# samplebot

Generate sound effects from text prompts with local open-weight models. CLI first, Python API underneath.

```bash
uv sync --extra gpu
uv run samplebot gen "heavy rain hitting a tin roof" -n music -n speech -m audioldm2 -s 5
uv run samplebot gen "door creak" -m audioldm2 --optimize    # prompt rewritten by a local Ollama LLM first
```

- Positive/negative keywords (`-p`, `-n`), seeds, variations, step count.
- Backends: `audioldm2` (ungated), `stable-audio` (needs HF token), `fake` (tests).
- Optional prompt optimization through Ollama, stdlib only.

Docs: [usage](docs/usage.md) · [plan](docs/plan.md) · [model survey](docs/start.md)
