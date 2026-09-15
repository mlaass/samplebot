import json
import threading
import urllib.request

from samplebot.cli import main
from samplebot.dashboard import list_runs, make_server


def test_dashboard_lists_and_serves_runs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    main(["gen", "Dog bark", "-n", "music", "-s", "0.1", "-c", "2"])
    (tmp_path / "out" / "orphan.json").write_text("{}")  # no wav -> ignored
    runs = list_runs(tmp_path / "out")
    assert [r["seed"] for r in runs] == [1, 0] and runs[0]["wav"] == "dog-bark-fake-1.wav"

    srv = make_server(tmp_path / "out", port=0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_port}"
    try:
        page = urllib.request.urlopen(f"{base}/").read()
        assert b"<title>samplebot runs" in page and b'id=playall' in page and b'onended="onEnded(this)"' in page
        api = json.load(urllib.request.urlopen(f"{base}/api/runs"))
        assert [r["negative"] for r in api] == [["music"], ["music"]]
        wav = urllib.request.urlopen(f"{base}/{api[0]['wav']}")
        assert wav.headers["Content-Type"].startswith("audio/") and wav.read()[:4] == b"RIFF"
    finally:
        srv.shutdown()


def test_make_server_falls_back_when_port_taken(tmp_path):
    a = make_server(tmp_path, port=0)
    b = make_server(tmp_path, port=a.server_port)  # taken -> free port, no crash
    assert b.server_port != a.server_port
    a.server_close(), b.server_close()
