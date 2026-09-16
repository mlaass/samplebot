import argparse
import json
import os
import sys
from pathlib import Path

from samplebot.core import BACKENDS, Prompt, batch, run


def build_parser():
    p = argparse.ArgumentParser(prog="samplebot", description="Generate sound effects from text with local models.")
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("gen", help="generate a sound")
    g.add_argument("prompt")
    g.add_argument("-p", "--positive", action="append", default=[], metavar="KW", help="extra positive keyword (repeatable)")
    g.add_argument("-n", "--negative", action="append", default=[], metavar="KW", help="negative keyword (repeatable)")
    g.add_argument("-m", "--model", default="fake", choices=list(BACKENDS))
    g.add_argument("-s", "--seconds", type=float, default=5.0)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--steps", type=int, default=0, help="inference steps, 0 = backend default")
    g.add_argument("--optimize", nargs="?", const="", metavar="LLM", help="rewrite the prompt with a local Ollama LLM first")
    g.add_argument("-o", "--out", type=Path, help="output .wav (default: out/<slug>-<seed>.wav)")
    g.add_argument("-c", "--count", type=int, default=1, help="number of variations (seed increments)")
    g.add_argument("--rate", type=int, metavar="HZ", help="resample the output to this sample rate (needs ffmpeg)")

    b = sub.add_parser("batch", help="run a JSON list of jobs, grouped by model so each model loads once")
    b.add_argument("jobs", help="jobs .json file, or - for stdin")
    b.add_argument("--rate", type=int, metavar="HZ", help="resample every output to this sample rate (needs ffmpeg)")
    b.add_argument("--force", action="store_true", help="regenerate jobs whose out file already exists")

    sub.add_parser("models", help="list generation backends")

    o = sub.add_parser("optimize", help="rewrite a prompt with a local LLM and print it")
    o.add_argument("prompt")
    o.add_argument("--llm", default=None)

    fe = sub.add_parser("fetch", help="download model repos into models/ with a resumable parallel downloader")
    fe.add_argument("repo", nargs="+", help="Hugging Face repo id, e.g. declare-lab/TangoFlux")
    fe.add_argument("-j", "--workers", type=int, default=12)
    fe.add_argument("-x", "--exclude", action="append", default=[], metavar="GLOB", help="skip files matching this glob (repeatable)")

    sv = sub.add_parser("serve", help="dashboard: browse and listen to previous runs")
    sv.add_argument("-d", "--dir", type=Path, default=Path("out"))
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=7333, help="default 7333; falls back to a free port if taken")
    return p


def cmd_gen(a) -> int:
    for path in run(Prompt(a.prompt, a.positive, a.negative), a.model, a.seconds, a.seed, a.steps, a.count, out=a.out, optimize=a.optimize,
                    rate=a.rate):
        print(path)
    return 0


def cmd_batch(a) -> int:
    jobs = json.loads(sys.stdin.read() if a.jobs == "-" else Path(a.jobs).read_text())
    if not isinstance(jobs, list):
        print("batch: jobs must be a JSON list", file=sys.stderr)
        return 2
    return 1 if batch(jobs, a.rate, a.force) else 0


def cmd_optimize(a) -> int:
    from samplebot.optimize import DEFAULT_LLM, optimize

    print(optimize(a.prompt, a.llm or DEFAULT_LLM))
    return 0


def cmd_fetch(a) -> int:
    from samplebot.fetch import fetch

    for repo in a.repo:
        print(fetch(repo, a.workers, exclude=tuple(a.exclude)))
    return 0


def cmd_serve(a) -> int:
    from samplebot.dashboard import serve

    serve(a.dir, a.host, a.port)
    return 0


def load_dotenv(path=Path(".env")):
    """KEY=VALUE lines into os.environ (existing vars win). Enough for HF_TOKEN; no dependency."""
    if path.exists():
        for line in path.read_text().splitlines():
            k, _, v = line.partition("=")
            if k.strip() and not k.startswith("#") and _:
                os.environ.setdefault(k.strip(), v.strip().strip("'\""))


def main(argv=None) -> int:
    load_dotenv()
    a = build_parser().parse_args(argv)
    if a.cmd == "models":
        print("\n".join(BACKENDS))
        return 0
    return {"gen": cmd_gen, "batch": cmd_batch, "optimize": cmd_optimize, "serve": cmd_serve, "fetch": cmd_fetch}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
