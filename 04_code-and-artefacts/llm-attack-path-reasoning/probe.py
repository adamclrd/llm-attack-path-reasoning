"""
Tests which models actually answer, and how much they cost per call.

The model listing and the models that are really served do not always agree: an
identifier can appear as supporting generateContent and still return 404 when
called. Rather than discovering that after a hundred wasted calls, this script
sends one real prompt to each candidate and reports what came back.

The prompt is a genuine scenario taken from the scenario file, so the latency
and token counts are the ones the campaign will actually see.

    python probe.py --provider gemini --models gemini-3.7-flash gemini-flash-latest
    python probe.py --provider gemini --all
"""

import argparse
import json
import time

from models import get_client, ModelError
from run import c1_prompt, parse_chain
from engine import verify_chain

# Candidates worth trying on the Gemini free tier, cheapest and most stable
# first. Names change between generations, so this is a starting point rather
# than a guarantee; --list-models in run.py shows what the key can address.
DEFAULT_GEMINI = [
    "gemini-flash-latest",
    "gemini-flash-lite-latest",
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-3-flash-preview",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="gemini")
    ap.add_argument("--models", nargs="+")
    ap.add_argument("--all", action="store_true",
                    help="try the built-in candidate list")
    ap.add_argument("--scenarios", default="scenarios_v2.json")
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--k", type=int, default=20)
    args = ap.parse_args()

    models = args.models or (DEFAULT_GEMINI if args.all else None)
    if not models:
        print("Pass --models <id> ... or --all")
        return

    with open(args.scenarios) as f:
        scenarios = json.load(f)
    pick = [s for s in scenarios
            if s["depth"] == args.depth and s.get("nominal_k") == args.k]
    if not pick:
        pick = scenarios
    scenario = pick[0]
    prompt = c1_prompt(scenario)
    call = get_client(args.provider)

    print(f"Probing on {scenario['id']} "
          f"(depth {scenario['depth']}, {scenario['distractors']} distractors, "
          f"~{len(prompt)//4} prompt tokens)\n")
    print(f"{'model':<34}{'status':<10}{'time':>8}{'in':>8}{'out':>7}  result")
    print("-" * 84)

    working = []
    for model in models:
        t0 = time.time()
        try:
            raw, i_tok, o_tok = call(prompt, model)
            elapsed = time.time() - t0
            chain, err = parse_chain(raw)
            if chain is None:
                note = f"unparsable ({err})"
            else:
                res = verify_chain(chain, scenario)
                note = (f"{len(chain)} steps, "
                        f"{'solved' if res['success'] else 'not solved'}")
            print(f"{model:<34}{'ok':<10}{elapsed:>7.1f}s{i_tok:>8}{o_tok:>7}"
                  f"  {note}")
            working.append((model, elapsed, o_tok))
        except Exception as exc:                          # noqa: BLE001
            elapsed = time.time() - t0
            msg = str(exc)
            code = "404" if "404" in msg else \
                   "429" if "429" in msg else \
                   "503" if "503" in msg else \
                   "timeout" if "timed out" in msg else "error"
            print(f"{model:<34}{code:<10}{elapsed:>7.1f}s{'-':>8}{'-':>7}"
                  f"  {msg[:34]}")
        time.sleep(4)

    if not working:
        print("\nNothing answered. Check the key, then run "
              "`python run.py --list-models`.")
        return

    print("\nUsable models, cheapest call first:")
    for model, elapsed, o_tok in sorted(working, key=lambda x: x[1]):
        print(f"  {model:<34}{elapsed:>6.1f}s  {o_tok} output tokens")

    best, elapsed, o_tok = min(working, key=lambda x: x[1])
    for label, calls in (("C1 over 160 instances", 160),
                         ("C1 over 72 instances", 72),
                         ("C2 over 36 instances (~12 turns each)", 430)):
        total = calls * (elapsed + 7)
        print(f"  {label:<40} ~{total/3600:.1f} h at --delay 7")
    if o_tok > 300:
        print("\nOutput is long for a short JSON answer, which suggests the "
              "model reasons internally before replying. A lite variant will "
              "be faster and will consume less of the daily quota.")


if __name__ == "__main__":
    main()
