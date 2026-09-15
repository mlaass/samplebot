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
