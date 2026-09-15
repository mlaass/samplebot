"""Runs inside vendor/moss-venv (its pins conflict with ours). One JSON request per stdin line, one JSON reply per stdout line.
Stdlib + torch + numpy + moss_soundeffect_v2 only; do not import samplebot here."""

import json
import os
import sys

import numpy as np
import torch
from moss_soundeffect_v2 import MossSoundEffectPipeline

proto = os.fdopen(os.dup(1), "w")  # keep the real stdout for the protocol...
sys.stdout = sys.stderr  # ...and send anything the pipeline prints to stderr

pipe = MossSoundEffectPipeline.from_pretrained(os.environ.get("SAMPLEBOT_MOSS", "OpenMOSS-Team/MOSS-SoundEffect-v2.0"),
                                               torch_dtype=torch.bfloat16, device="cuda" if torch.cuda.is_available() else "cpu")
proto.write(json.dumps({"ready": True, "sr": pipe.sample_rate}) + "\n")
proto.flush()

for line in sys.stdin:
    req = json.loads(line)
    try:
        audio = pipe(prompt=req["prompt"], negative_prompt=req.get("negative") or "", seconds=float(req["seconds"]),
                     num_inference_steps=int(req["steps"]), cfg_scale=float(req.get("cfg", 4.0)), seed=int(req["seed"]))
        np.save(req["out"], audio[0].float().cpu().numpy())  # (C, T)
        reply = {"ok": True, "sr": pipe.sample_rate}
    except Exception as e:  # report, keep serving
        reply = {"error": f"{type(e).__name__}: {e}"}
    proto.write(json.dumps(reply) + "\n")
    proto.flush()
