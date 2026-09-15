import io
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

import samplebot.fetch as fetch

BLOB = bytes(range(256)) * 4000  # 1 MB, not a multiple of the chunk size below


class RangeHandler(BaseHTTPRequestHandler):
    fail_first = set()

    def do_GET(self):
        if self.path.startswith("/api/models/"):
            body = json.dumps({"siblings": [{"rfilename": "unet/w.safetensors", "size": len(BLOB)}, {"rfilename": "unet/pytorch_model.bin", "size": 1}, {"rfilename": "unet/flax_model.msgpack", "size": 1},
                                            {"rfilename": "config.json", "size": 2}]}).encode()
            return self._send(200, body)
        if self.path.endswith("config.json"):
            return self._send(200, b"{}")
        rng = self.headers.get("Range", "bytes=0-").removeprefix("bytes=")
        start, end = (int(x) for x in rng.split("-"))
        if start in RangeHandler.fail_first:  # simulate one throttled chunk
            RangeHandler.fail_first.discard(start)
            return self._send(429, b"no permits")
        self.send_response(206)
        self.send_header("Content-Length", str(end - start + 1))
        self.end_headers()
        self.wfile.write(BLOB[start : end + 1])

    def _send(self, code, body):
        self.send_response(code)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


@pytest.fixture
def hub(monkeypatch, tmp_path):
    srv = ThreadingHTTPServer(("127.0.0.1", 0), RangeHandler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    monkeypatch.setattr(fetch, "HF", f"http://127.0.0.1:{srv.server_port}")
    monkeypatch.setattr(fetch, "MODELS", tmp_path / "models")
    monkeypatch.setattr(fetch, "CHUNK", 300_000)
    monkeypatch.setattr(fetch.time, "sleep", lambda s: None)
    yield srv
    srv.shutdown()


def test_fetch_parallel_ranges_with_retry_and_skips_duplicate_bin(hub, tmp_path):
    RangeHandler.fail_first = {300_000}
    dest = fetch.fetch("acme/model", workers=4, log=io.StringIO())
    assert (dest / "unet/w.safetensors").read_bytes() == BLOB
    assert not (dest / "unet/pytorch_model.bin").exists() and not (dest / "unet/flax_model.msgpack").exists() and (dest / "config.json").exists()
    assert not list(dest.rglob("*.part")) and not list(dest.rglob("*.done"))
    assert fetch.model_path("acme/model") == str(dest) and fetch.model_path("acme/other") == "acme/other"


def test_fetch_resumes_partial(hub, tmp_path):
    dest = tmp_path / "models/acme/model/unet/w.safetensors"
    dest.parent.mkdir(parents=True)
    part = dest.with_name("w.safetensors.part")
    part.write_bytes(b"\0" * len(BLOB))
    dest.with_name("w.safetensors.done").write_text("[0]")  # pretend chunk 0 is done (its bytes are zeros)
    fetch.download(f"{fetch.HF}/acme/model/resolve/main/unet/w.safetensors", dest, len(BLOB), workers=2, log=io.StringIO())
    data = dest.read_bytes()
    assert data[:300_000] == b"\0" * 300_000 and data[300_000:] == BLOB[300_000:]
