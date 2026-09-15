import json
from unittest import mock

from samplebot.optimize import optimize


def fake_urlopen(reply):
    resp = mock.MagicMock()
    resp.read.return_value = json.dumps({"message": {"content": reply}}).encode()
    resp.__enter__.return_value = resp
    return mock.patch("urllib.request.urlopen", return_value=resp)


def test_optimize_strips_think_and_quotes():
    with fake_urlopen('<think>hmm</think>\n"Heavy rain drumming on a corrugated tin roof, close-up"\nextra') as m:
        assert optimize("rain on roof", "qwen3:14b") == "Heavy rain drumming on a corrugated tin roof, close-up"
    body = json.loads(m.call_args.args[0].data)
    assert body["model"] == "qwen3:14b" and body["messages"][1]["content"] == "rain on roof"
