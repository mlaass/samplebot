"""Browse, listen, and generate. Stdlib only: serves out/ plus one HTML page and a small JSON API."""

import functools
import json
import threading
import traceback
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from samplebot.core import BACKENDS, Prompt, run, unload

DEFAULT_PORT = 7333
GEN_LOCK = threading.Lock()  # ponytail: one generation at a time, it is one GPU

PAGE = """<!doctype html><meta charset=utf-8><title>samplebot runs</title>
<style>
body{font:14px system-ui;margin:1.5rem;background:#111;color:#ddd}h1{font-size:1.2rem}
input,select,textarea{background:#222;color:#eee;border:1px solid #444;padding:.4rem;font:inherit}
#q{width:100%;margin:1rem 0}form{display:grid;grid-template-columns:repeat(6,1fr);gap:.5rem;max-width:1100px}
form label{display:flex;flex-direction:column;gap:.2rem;color:#aaa;font-size:12px}.wide{grid-column:span 6}.w3{grid-column:span 3}
form textarea{min-height:2.6em}#status{color:#fc6;min-height:1.2em}#err{color:#f66;white-space:pre-wrap}
table{border-collapse:collapse;width:100%}td,th{padding:.4rem .6rem;border-bottom:1px solid #333;text-align:left;vertical-align:top}
th{color:#999;font-weight:normal}code{color:#9cf}.neg{color:#f99}.src{color:#888}audio{width:260px}
button{background:#333;color:#eee;border:1px solid #555;padding:.4rem .8rem;cursor:pointer}button.go{background:#264}
button:disabled{opacity:.5;cursor:wait}.chk{flex-direction:row!important;align-items:center}tr.playing{background:#1e2a1e}
</style>
<h1>samplebot</h1>
<form id=f onsubmit="return go(event)">
<label class=wide>prompt<textarea name=text required></textarea></label>
<label class=w3>positive keywords (comma separated)<input name=positive></label>
<label class=w3>negative keywords (comma separated)<input name=negative></label>
<label>model<select name=model id=model></select></label>
<label>seconds<input name=seconds type=number step=0.5 min=0.5 value=5></label>
<label>seed<input name=seed type=number value=0></label>
<label>steps (0 = default)<input name=steps type=number min=0 value=0></label>
<label>count<input name=count type=number min=1 value=1></label>
<label class=chk><input name=optimize type=checkbox> optimize with LLM</label>
<label class=w3>ollama model (for optimize)<input name=llm placeholder="default"></label>
<div class=w3 style="align-self:end"><button class=go id=gobtn>generate</button> <span id=status></span></div>
<div class=wide id=err></div>
</form>
<input id=q placeholder="filter by prompt, model, keyword…">
<p><button id=playall>▶ play all</button><label><input type=checkbox id=auto checked> autoplay next</label></p>
<table><thead><tr><th>when</th><th>play</th><th>prompt</th><th>model</th><th>s</th><th>seed</th><th>steps</th><th>file</th><th></th></tr></thead><tbody id=rows></tbody></table>
<script>
let runs=[];const $=id=>document.getElementById(id);const esc=s=>String(s??'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
function render(){const q=$('q').value.toLowerCase();
$('rows').innerHTML=runs.filter(r=>JSON.stringify(r).toLowerCase().includes(q)).map((r,i)=>`<tr>
<td>${new Date(r.mtime*1000).toLocaleString()}</td><td><audio controls preload=none src="${esc(r.wav)}" onplay="onPlay(this)" onended="onEnded(this)"></audio></td>
<td>${esc(r.text)}${r.source&&r.source!==r.text?`<br><span class=src>from: ${esc(r.source)}</span>`:''}
${(r.positive||[]).length?`<br>+ ${esc(r.positive.join(', '))}`:''}${(r.negative||[]).length?`<br><span class=neg>− ${esc(r.negative.join(', '))}</span>`:''}</td>
<td>${esc(r.model)}</td><td>${esc(r.seconds)}</td><td>${esc(r.seed)}</td><td>${esc(r.steps)}</td><td><code>${esc(r.wav)}</code></td>
<td><button type=button onclick="reuse(${i})" title="load this run's prompt and params into the form">reuse</button>
<button type=button onclick="reuse(${i},1)" title="same params, next seed">variation</button></td></tr>`).join('')}
function reuse(i,next){const r=runs.filter(r=>JSON.stringify(r).toLowerCase().includes($('q').value.toLowerCase()))[i],f=$('f');
f.text.value=r.text;f.positive.value=(r.positive||[]).join(', ');f.negative.value=(r.negative||[]).join(', ');
if([...f.model.options].some(o=>o.value===r.model))f.model.value=r.model;f.seconds.value=r.seconds;f.seed.value=(r.seed||0)+(next?1:0);f.steps.value=r.steps||0;
f.optimize.checked=false;window.scrollTo(0,0);if(next)go();else f.text.focus()}
async function go(e){if(e)e.preventDefault();const f=$('f'),split=s=>s.split(',').map(x=>x.trim()).filter(Boolean);
const body={text:f.text.value,positive:split(f.positive.value),negative:split(f.negative.value),model:f.model.value,seconds:+f.seconds.value,
seed:+f.seed.value,steps:+f.steps.value,count:+f.count.value,optimize:f.optimize.checked?(f.llm.value||''):null};
$('gobtn').disabled=true;$('status').textContent='generating…';$('err').textContent='';
try{const r=await fetch('api/generate',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify(body)});const j=await r.json();
if(!r.ok)throw new Error(j.error);$('status').textContent=`done: ${j.paths.join(', ')}`;await load();
const a=document.querySelector('audio');if(a)a.play()}catch(err){$('status').textContent='';$('err').textContent=err.message}
$('gobtn').disabled=false;return false}
const players=()=>[...document.querySelectorAll('audio')];
function onPlay(a){players().forEach(o=>{if(o!==a)o.pause();o.closest('tr').classList.toggle('playing',o===a)})}
function onEnded(a){a.closest('tr').classList.remove('playing');if(!$('auto').checked)return;const ps=players(),n=ps[ps.indexOf(a)+1];if(n)n.play()}
$('playall').onclick=()=>{const p=players()[0];if(p){$('auto').checked=true;p.play()}};
async function load(){runs=await (await fetch('api/runs')).json();render()}
fetch('api/models').then(r=>r.json()).then(m=>{$('model').innerHTML=m.map(x=>`<option>${x}</option>`).join('');if(m.includes('audioldm2'))$('model').value='audioldm2'});
load();$('q').oninput=render;
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


def api_generate(directory: Path, req: dict) -> dict:
    """Raises ValueError for a bad request (-> 400). `out` is a .wav path relative to the served directory and must stay inside it."""
    directory = Path(directory).resolve()
    prompt = Prompt(str(req.get("text", "")).strip(), list(req.get("positive", [])), list(req.get("negative", [])))
    if not prompt.text:
        raise ValueError("prompt is empty")
    count = max(1, int(req.get("count", 1)))
    out = None
    if req.get("out") is not None:
        out = (directory / str(req["out"])).resolve()  # resolves "..", symlinks and absolute paths before the containment check
        if not out.is_relative_to(directory) or out.suffix != ".wav":
            raise ValueError(f"out must be a .wav path inside the served directory: {req['out']!r}")
        if count != 1:
            raise ValueError("out needs count 1")
    rate = int(req["rate"]) if req.get("rate") else None
    with GEN_LOCK:
        paths = run(prompt, req.get("model", "fake"), float(req.get("seconds", 5)), int(req.get("seed", 0)), int(req.get("steps", 0)),
                    count, out_dir=directory, out=out, optimize=req.get("optimize"), rate=rate)
    return {"paths": [p.relative_to(directory).as_posix() for p in paths]}


class Handler(SimpleHTTPRequestHandler):
    def do_GET(self):
        path = self.path.split("?")[0]
        if path in ("/", "/index.html"):
            return self._send(PAGE.encode(), "text/html; charset=utf-8")
        if path == "/api/runs":
            return self._json(list_runs(Path(self.directory)))
        if path == "/api/models":
            return self._json(list(BACKENDS))
        return super().do_GET()

    def do_POST(self):
        if self.path == "/api/unload":
            with GEN_LOCK:  # waits for a running generation instead of pulling the model out from under it
                return self._json({"unloaded": unload()})
        if self.path != "/api/generate":
            return self.send_error(404)
        try:
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0)) or b"{}"))
            return self._json(api_generate(Path(self.directory), req))
        except Exception as e:  # surface the real error in the UI
            traceback.print_exc()
            return self._json({"error": f"{type(e).__name__}: {e}"}, 400 if isinstance(e, ValueError) else 500)

    def _json(self, obj, status=200):
        self._send(json.dumps(obj).encode(), "application/json", status)

    def _send(self, body: bytes, ctype: str, status=200):
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):  # quiet
        pass


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
    Path(directory).mkdir(parents=True, exist_ok=True)
    srv = make_server(directory, host, port)
    print(f"samplebot dashboard: http://{host}:{srv.server_port}/  (serving {Path(directory).resolve()})", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
