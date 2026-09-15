import argparse
import sys
from pathlib import Path

from samplebot.core import BACKENDS, Prompt, run


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

    sub.add_parser("models", help="list generation backends")

    o = sub.add_parser("optimize", help="rewrite a prompt with a local LLM and print it")
    o.add_argument("prompt")
    o.add_argument("--llm", default=None)

    fe = sub.add_parser("fetch", help="download model repos into models/ with a resumable parallel downloader")
    fe.add_argument("repo", nargs="+", help="Hugging Face repo id, e.g. declare-lab/TangoFlux")
    fe.add_argument("-j", "--workers", type=int, default=12)

    sv = sub.add_parser("serve", help="dashboard: browse and listen to previous runs")
    sv.add_argument("-d", "--dir", type=Path, default=Path("out"))
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=7333, help="default 7333; falls back to a free port if taken")
    return p


def cmd_gen(a) -> int:
    for path in run(Prompt(a.prompt, a.positive, a.negative), a.model, a.seconds, a.seed, a.steps, a.count, out=a.out, optimize=a.optimize):
        print(path)
    return 0


def cmd_optimize(a) -> int:
    from samplebot.optimize import DEFAULT_LLM, optimize

    print(optimize(a.prompt, a.llm or DEFAULT_LLM))
    return 0


def cmd_fetch(a) -> int:
    from samplebot.fetch import fetch

    for repo in a.repo:
        print(fetch(repo, a.workers))
    return 0


def cmd_serve(a) -> int:
    from samplebot.dashboard import serve

    serve(a.dir, a.host, a.port)
    return 0


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    if a.cmd == "models":
        print("\n".join(BACKENDS))
        return 0
    return {"gen": cmd_gen, "optimize": cmd_optimize, "serve": cmd_serve, "fetch": cmd_fetch}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
