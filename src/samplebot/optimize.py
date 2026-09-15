"""Rewrite a prompt with a local Ollama LLM. Stdlib only."""

import json
import os
import re
import urllib.request

OLLAMA = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
DEFAULT_LLM = os.environ.get("SAMPLEBOT_LLM", "qwen2.5:7b-instruct")

SYSTEM = """You rewrite short sound-effect requests into prompts for text-to-audio diffusion models (AudioLDM2, Stable Audio Open).
Rules: one line, at most 40 words. Describe the sound source, material, action, environment and perspective concretely
(e.g. "close-up", "distant", "reverberant hall"). Keep every detail the user gave. No speech, no music unless asked.
Output only the rewritten prompt, nothing else."""


def optimize(prompt: str, llm: str = DEFAULT_LLM, host: str = OLLAMA) -> str:
    body = {
        "model": llm,
        "stream": False,
        "think": False,
        "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
    }
    req = urllib.request.Request(f"{host}/api/chat", json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        text = json.load(r)["message"]["content"]
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)  # qwen3 without think:false support
    return text.strip().splitlines()[0].strip().strip('"')


def list_llms(host: str = OLLAMA) -> list[str]:
    with urllib.request.urlopen(f"{host}/api/tags", timeout=5) as r:
        return [m["name"] for m in json.load(r)["models"]]
