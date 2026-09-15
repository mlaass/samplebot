import json
import wave

import numpy as np
import pytest

from samplebot import Prompt, generate, write_wav
from samplebot.cli import main


def test_prompt_composition():
    p = Prompt("rain on a tin roof", positive=["heavy", " close-up "], negative=["music", "speech"])
    assert p.positive_text() == "rain on a tin roof, heavy, close-up"
    assert p.negative_text() == "music, speech"
    assert Prompt("x").negative_text() == ""


def test_fake_backend_is_deterministic():
    a, sr = generate(Prompt("dog bark"), "fake", seconds=0.5, seed=1)
    b, _ = generate(Prompt("dog bark"), "fake", seconds=0.5, seed=1)
    c, _ = generate(Prompt("dog bark"), "fake", seconds=0.5, seed=2)
    assert a.shape == (1, sr // 2) and a.dtype == np.float32
    assert np.array_equal(a, b) and not np.array_equal(a, c)


def test_unknown_model():
    with pytest.raises(ValueError, match="unknown model"):
        generate(Prompt("x"), "nope")


def test_write_wav_roundtrip(tmp_path):
    audio = np.stack([np.linspace(-1, 1, 100), np.zeros(100)]).astype(np.float32)
    path = write_wav(tmp_path / "a" / "b.wav", audio, 22050, {"k": 1})
    with wave.open(str(path)) as w:
        assert (w.getnchannels(), w.getframerate(), w.getnframes(), w.getsampwidth()) == (2, 22050, 100, 2)
        frames = np.frombuffer(w.readframes(100), "<i2").reshape(100, 2)
    assert frames[0, 0] == -32767 and frames[-1, 0] == 32767 and frames[:, 1].sum() == 0
    assert json.loads(path.with_suffix(".json").read_text()) == {"k": 1}


def test_write_wav_scales_down_overshoot(tmp_path):
    path = write_wav(tmp_path / "loud.wav", np.array([[2.0, -1.0, 0.5]], dtype=np.float32), 8000)
    with wave.open(str(path)) as w:
        assert list(np.frombuffer(w.readframes(3), "<i2")) == [32767, -16383, 8191]


def test_cli_gen(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert main(["gen", "Wind howling", "-n", "music", "-s", "0.2", "-c", "2"]) == 0
    outs = capsys.readouterr().out.split()
    assert outs == ["out/wind-howling-fake-0.wav", "out/wind-howling-fake-1.wav"]
    meta = json.loads((tmp_path / "out/wind-howling-fake-1.json").read_text())
    assert meta["negative"] == ["music"] and meta["seed"] == 1 and meta["model"] == "fake"


def test_cli_models(capsys):
    main(["models"])
    assert "fake" in capsys.readouterr().out.split()


def test_switching_models_unloads_previous(monkeypatch):
    import sys
    import types

    import samplebot.core as core

    calls = []
    other = types.ModuleType("samplebot.backends.other")
    other.generate = lambda *a: (np.zeros((1, 8), np.float32), 8000)
    other.unload = lambda: calls.append("other")
    monkeypatch.setitem(sys.modules, "samplebot.backends.other", other)
    monkeypatch.setitem(core.BACKENDS, "other", "samplebot.backends.other")
    import samplebot.backends.fake as fake

    monkeypatch.setattr(fake, "unload", lambda: calls.append("fake"))
    monkeypatch.setattr(core, "_loaded", None)
    generate(Prompt("a"), "fake", 0.01)
    generate(Prompt("a"), "fake", 0.01)  # same model: no unload
    generate(Prompt("a"), "other", 0.01)  # switch: fake unloaded
    generate(Prompt("a"), "fake", 0.01)  # switch back: other unloaded
    assert calls == ["fake", "other"]


def test_all_backends_import_without_torch_loaded():
    import importlib
    import sys

    from samplebot.core import BACKENDS

    for mod in BACKENDS.values():
        m = importlib.import_module(mod)
        assert callable(m.generate) and callable(m.unload)
    assert "torch" not in sys.modules  # heavy imports stay inside generate()/pipe()


def test_cli_gen_optimize_bakes_keywords(tmp_path, monkeypatch):
    import samplebot.optimize

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(samplebot.optimize, "optimize", lambda text, llm: f"LLM({text})")
    main(["gen", "wind", "-p", "gusty", "-n", "music", "-s", "0.1", "--optimize", "x", "-o", "w.wav"])
    meta = json.loads((tmp_path / "w.json").read_text())
    assert meta["text"] == "LLM(wind, gusty)" and meta["positive"] == [] and meta["negative"] == ["music"] and meta["source"] == "wind"


def test_moss_backend_reports_missing_venv(monkeypatch):
    import samplebot.backends.moss as moss

    monkeypatch.setattr(moss, "PYTHON", "/nonexistent/python")
    moss.worker.cache_clear()
    with pytest.raises(RuntimeError, match="create the MOSS venv"):
        moss.generate("x", "", 1, 0, 0)


def test_moss_worker_compiles():
    import py_compile
    from samplebot.backends.moss import WORKER

    py_compile.compile(str(WORKER), doraise=True)


def test_load_dotenv(tmp_path, monkeypatch):
    from samplebot.cli import load_dotenv

    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setenv("KEEP", "orig")
    f = tmp_path / ".env"
    f.write_text("# comment\nHF_TOKEN='hf_x'\nKEEP=new\nBROKEN\n")
    load_dotenv(f)
    import os

    assert os.environ["HF_TOKEN"] == "hf_x" and os.environ["KEEP"] == "orig"
