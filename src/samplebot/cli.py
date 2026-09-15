import argparse
import re
import sys
from pathlib import Path

from samplebot.core import BACKENDS, Prompt, generate, prompt_meta, write_wav


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:60] or "sound"


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
    return p


def cmd_gen(a) -> int:
    prompt = Prompt(a.prompt, a.positive, a.negative)
    if a.optimize is not None:
        from samplebot.optimize import DEFAULT_LLM, optimize

        # keywords go into the LLM too, so they are baked into the rewritten text and not appended twice
        prompt = Prompt(optimize(prompt.positive_text(), a.optimize or DEFAULT_LLM), negative=a.negative)
        print(f"optimized prompt: {prompt.text}", file=sys.stderr)
    for i in range(a.count):
        seed = a.seed + i
        audio, sr = generate(prompt, a.model, a.seconds, seed, a.steps)
        out = a.out or Path("out") / f"{slug(prompt.text)}-{a.model}-{seed}.wav"
        if a.out and a.count > 1:
            out = a.out.with_stem(f"{a.out.stem}-{seed}")
        write_wav(out, audio, sr, prompt_meta(prompt, source=a.prompt, model=a.model, seconds=a.seconds, seed=seed, steps=a.steps, sample_rate=sr))
        print(out)
    return 0


def cmd_optimize(a) -> int:
    from samplebot.optimize import DEFAULT_LLM, optimize

    print(optimize(a.prompt, a.llm or DEFAULT_LLM))
    return 0


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    if a.cmd == "models":
        print("\n".join(BACKENDS))
        return 0
    return {"gen": cmd_gen, "optimize": cmd_optimize}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
