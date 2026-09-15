"""Browse and listen to previous runs. Stdlib only: serves out/ plus one HTML page and a JSON index."""

import functools
import json
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PAGE = """<!doctype html><meta charset=utf-8><title>samplebot runs</title>
<style>
body{font:14px system-ui;margin:1.5rem;background:#111;color:#ddd}h1{font-size:1.2rem}
input{width:100%;padding:.5rem;background:#222;color:#eee;border:1px solid #444;margin-bottom:1rem}
table{border-collapse:collapse;width:100%}td,th{padding:.4rem .6rem;border-bottom:1px solid #333;text-align:left;vertical-align:top}
th{color:#999;font-weight:normal}code{color:#9cf}.neg{color:#f99}.src{color:#888}audio{width:260px}
</style>
<h1>samplebot runs</h1><input id=q placeholder="filter by prompt, model, keyword…" autofocus>
<table><thead><tr><th>when</th><th>play</th><th>prompt</th><th>model</th><th>s</th><th>seed</th><th>steps</th><th>file</th></tr></thead><tbody id=rows></tbody></table>
<script>
let runs=[];const esc=s=>String(s??'').replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
function render(){const q=document.getElementById('q').value.toLowerCase();
document.getElementById('rows').innerHTML=runs.filter(r=>JSON.stringify(r).toLowerCase().includes(q)).map(r=>`<tr>
<td>${new Date(r.mtime*1000).toLocaleString()}</td><td><audio controls preload=none src="${esc(r.wav)}"></audio></td>
<td>${esc(r.text)}${r.source&&r.source!==r.text?`<br><span class=src>from: ${esc(r.source)}</span>`:''}
${(r.positive||[]).length?`<br>+ ${esc(r.positive.join(', '))}`:''}${(r.negative||[]).length?`<br><span class=neg>− ${esc(r.negative.join(', '))}</span>`:''}</td>
<td>${esc(r.model)}</td><td>${esc(r.seconds)}</td><td>${esc(r.seed)}</td><td>${esc(r.steps)}</td><td><code>${esc(r.wav)}</code></td></tr>`).join('')}
fetch('api/runs').then(r=>r.json()).then(j=>{runs=j;render()});document.getElementById('q').oninput=render;
</script>"""


def list_runs(directory: Path) -> list[dict]:
    """Every .json sidecar with a sibling .wav, newest first."""
    runs = []
    for meta in Path(directory).rglob("*.json"):
        wav = meta.with_suffix(".wav")
        if not wav.exists():
            continue
        try:
            d = json.loads(meta.read_text())
        except ValueError:
            continue
        runs.append({**d, "wav": wav.relative_to(directory).as_posix(), "mtime": wav.stat().st_mtime})
    return sorted(runs, key=lambda r: (r["mtime"], r["wav"]), reverse=True)  # name breaks mtime ties


class Handler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path.split("?")[0] in ("/", "/index.html"):
            return self._send(PAGE.encode(), "text/html; charset=utf-8")
        if self.path == "/api/runs":
            return self._send(json.dumps(list_runs(Path(self.directory))).encode(), "application/json")
        return super().do_GET()

    def _send(self, body: bytes, ctype: str):
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):  # quiet
        pass


DEFAULT_PORT = 7333


def make_server(directory: Path, host: str = "127.0.0.1", port: int = DEFAULT_PORT) -> ThreadingHTTPServer:
    handler = functools.partial(Handler, directory=str(directory))
    try:
        return ThreadingHTTPServer((host, port), handler)
    except OSError as e:
        if port == 0:
            raise
        print(f"port {port} in use ({e.strerror}), picking a free one", flush=True)
        return ThreadingHTTPServer((host, 0), handler)


def serve(directory: Path, host: str = "127.0.0.1", port: int = DEFAULT_PORT):
    srv = make_server(directory, host, port)
    print(f"samplebot dashboard: http://{host}:{srv.server_port}/  (serving {Path(directory).resolve()})", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
