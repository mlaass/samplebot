import json
import shutil
import threading
import urllib.error
import urllib.request

import pytest

from samplebot.cli import main
from samplebot.dashboard import list_runs, make_server


@pytest.fixture
def server(tmp_path):
    srv = make_server(tmp_path / "out", port=0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()


def post(url, body):
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req))


def test_dashboard_lists_and_serves_runs(tmp_path, monkeypatch, server):
    monkeypatch.chdir(tmp_path)
    main(["gen", "Dog bark", "-n", "music", "-s", "0.1", "-c", "2"])
    (tmp_path / "out" / "orphan.json").write_text("{}")  # no wav -> ignored
    runs = list_runs(tmp_path / "out")
    assert [r["seed"] for r in runs] == [1, 0] and runs[0]["wav"] == "dog-bark-fake-1.wav"

    page = urllib.request.urlopen(f"{server}/").read()
    assert b"<title>samplebot runs" in page and b"id=playall" in page and b'onended="onEnded(this)"' in page
    api = json.load(urllib.request.urlopen(f"{server}/api/runs"))
    assert [r["negative"] for r in api] == [["music"], ["music"]]
    wav = urllib.request.urlopen(f"{server}/{api[0]['wav']}")
    assert wav.headers["Content-Type"].startswith("audio/") and wav.read()[:4] == b"RIFF"
    assert "fake" in json.load(urllib.request.urlopen(f"{server}/api/models"))


def test_api_generate(tmp_path, server):
    out = post(f"{server}/api/generate", {"text": "Cat meow", "negative": ["dog"], "model": "fake", "seconds": 0.1, "seed": 4, "count": 2})
    assert out["paths"] == ["cat-meow-fake-4.wav", "cat-meow-fake-5.wav"]
    runs = json.load(urllib.request.urlopen(f"{server}/api/runs"))
    assert [(r["seed"], r["negative"]) for r in runs] == [(5, ["dog"]), (4, ["dog"])]

    with pytest.raises(urllib.error.HTTPError) as e:
        post(f"{server}/api/generate", {"text": "x", "model": "nope"})
    assert e.value.code == 400 and "unknown model" in json.load(e.value)["error"]
    with pytest.raises(urllib.error.HTTPError) as e:
        post(f"{server}/api/generate", {"text": "  "})
    assert "empty" in json.load(e.value)["error"]


def test_api_generate_out_and_rate(tmp_path, server):
    out = post(f"{server}/api/generate", {"text": "axe", "model": "fake", "seconds": 0.1, "seed": 2, "out": "axe_chop/r2-ui1-fake-2.wav"})
    assert out["paths"] == ["axe_chop/r2-ui1-fake-2.wav"] and (tmp_path / "out/axe_chop/r2-ui1-fake-2.wav").exists()
    if shutil.which("ffmpeg"):
        post(f"{server}/api/generate", {"text": "axe", "model": "fake", "seconds": 0.1, "rate": 44100, "out": "r.wav"})
        assert json.loads((tmp_path / "out/r.json").read_text())["sample_rate"] == 44100

    for bad in ({"out": "../evil.wav"}, {"out": str(tmp_path / "abs.wav")}, {"out": "a/../../x.wav"}, {"out": "x.txt"},
                {"out": "two.wav", "count": 2}):
        with pytest.raises(urllib.error.HTTPError) as e:
            post(f"{server}/api/generate", {"text": "axe", "model": "fake", "seconds": 0.1, **bad})
        assert e.value.code == 400, bad
    assert not (tmp_path / "evil.wav").exists() and not (tmp_path / "abs.wav").exists() and not (tmp_path / "x.wav").exists()


def test_make_server_falls_back_when_port_taken(tmp_path):
    a = make_server(tmp_path, port=0)
    b = make_server(tmp_path, port=a.server_port)  # taken -> free port, no crash
    assert b.server_port != a.server_port
    a.server_close(), b.server_close()
