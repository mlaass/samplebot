"""Resumable parallel downloader for Hugging Face repos. Stdlib only.

The HF CDN throttles each connection (~250 KB/s here) but allows many; hf_xet and hf_transfer both fell over on it,
so this fetches fixed-size ranges with a thread pool, writes them in place, and records finished chunks so an
interrupted run resumes. Files land in models/<repo>/ with the repo layout, which from_pretrained accepts as-is.
"""

import json
import os
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HF = os.environ.get("HF_ENDPOINT", "https://huggingface.co")
MODELS = Path(os.environ.get("SAMPLEBOT_MODELS", "models"))
CHUNK = 16 << 20


def model_path(repo: str) -> str:
    """Local models/<repo> if fetched, else the hub id (from_pretrained downloads it)."""
    local = MODELS / repo
    return str(local) if (local / "config.json").exists() or (local / "model_index.json").exists() else repo


def _open(url: str, headers: dict, timeout=60):
    req = urllib.request.Request(url, headers={**headers, "User-Agent": "samplebot"})
    if tok := os.environ.get("HF_TOKEN"):
        req.add_header("Authorization", f"Bearer {tok}")
    return urllib.request.urlopen(req, timeout=timeout)


def list_files(repo: str) -> list[tuple[str, int]]:
    with _open(f"{HF}/api/models/{repo}?blobs=true", {}) as r:
        sib = json.load(r)["siblings"]
    names = {s["rfilename"] for s in sib}
    keep = []
    for s in sib:
        n = s["rfilename"]
        if n.endswith(".bin") and n[:-4] + ".safetensors" in names:
            continue  # duplicate weights
        keep.append((n, int(s.get("size") or 0)))
    return keep


def download(url: str, dest: Path, size: int, workers: int = 12, log=sys.stderr) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size == size:
        return dest
    part, donefile = dest.with_name(dest.name + ".part"), dest.with_name(dest.name + ".done")
    done = set(json.loads(donefile.read_text())) if donefile.exists() and part.exists() else set()
    with open(part, "ab") as f:
        f.truncate(size)
    fd = os.open(part, os.O_WRONLY)
    lock, todo, t0 = threading.Lock(), [i for i in range(-(-size // CHUNK)) if i not in done], time.time()
    got = [len(done) * CHUNK]

    def grab(i):
        start, end = i * CHUNK, min(size, (i + 1) * CHUNK) - 1
        for attempt in range(8):
            try:
                with _open(url, {"Range": f"bytes={start}-{end}"}) as r:
                    data = r.read()
                if len(data) != end - start + 1:
                    raise OSError(f"short read {len(data)}")
                os.pwrite(fd, data, start)
                with lock:
                    done.add(i)
                    got[0] += len(data)
                    donefile.write_text(json.dumps(sorted(done)))
                    print(f"\r{dest.name}: {got[0] / 1e6:.0f}/{size / 1e6:.0f} MB {got[0] / 1e6 / max(time.time() - t0, 1e-3):.1f} MB/s", end="", file=log, flush=True)
                return
            except Exception as e:  # throttled / reset: back off and retry the chunk
                time.sleep(min(2**attempt, 30))
                err = e
        raise RuntimeError(f"chunk {i} of {dest.name} failed: {err}")

    try:
        with ThreadPoolExecutor(workers) as ex:
            list(ex.map(grab, todo))
    finally:
        os.close(fd)
    print(file=log)
    part.rename(dest)
    donefile.unlink(missing_ok=True)
    return dest


def fetch(repo: str, workers: int = 12, log=sys.stderr) -> Path:
    dest = MODELS / repo
    for name, size in list_files(repo):
        download(f"{HF}/{repo}/resolve/main/{name}", dest / name, size, workers, log)
    return dest
