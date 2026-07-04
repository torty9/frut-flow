#!/usr/bin/env python3
"""
try_repair.py — eyeball the accuracy of the on-device context repair (cleanup="local").

  RUN THE SUITE:   frutflow/.venv/bin/python frutflow/try_repair.py
  TRY ONE PHRASE:  frutflow/.venv/bin/python frutflow/try_repair.py "meet at the peer at noon"
  TRY A BIGGER MODEL:
      REPAIR_MODEL=mlx-community/Qwen2.5-3B-Instruct-4bit \
          frutflow/.venv/bin/python frutflow/try_repair.py

Each case is  (kind, input_as_the_recognizer_heard_it, intended_correct_text):
  fix   the recognizer misheard a word   -> PASS if the repair yields `intended`
  keep  already correct                  -> PASS if the WORDS are left unchanged
  ask   a question / command             -> PASS if it's returned, NOT answered

Scoring ignores capitalization + edge punctuation, so it grades WORDS, not style
(e.g. "its" vs "it's" still counts as different — that's the whole point).
Add your own lines to CASES and re-run.
"""

CASES = [
    # --- should FIX (context makes the misheard word obvious) --------------------
    ("fix",  "meet me at the peer at noon",                 "meet me at the pier at noon"),
    ("fix",  "put the boxes over they're by the door",      "put the boxes over there by the door"),
    ("fix",  "the server lost it's connection",             "the server lost its connection"),
    ("fix",  "have them Ryan Opus for the review",          "have them run Opus for the review"),
    ("fix",  "your going to love this feature",             "you're going to love this feature"),
    ("fix",  "we should of tested it first",                "we should have tested it first"),
    ("fix",  "there's a whole in the wall",                 "there's a hole in the wall"),
    ("fix",  "can you send me they're address",             "can you send me their address"),
    ("fix",  "its to cold to go outside",                   "it's too cold to go outside"),
    ("fix",  "I want to by two new monitors",               "I want to buy two new monitors"),
    ("fix",  "the affect was immediate",                    "the effect was immediate"),

    # --- should KEEP (already correct — must NOT touch the words) ----------------
    ("keep", "the API call is asynchronous and returns a promise",
             "the API call is asynchronous and returns a promise"),
    ("keep", "I read that book last night and it was great",
             "I read that book last night and it was great"),
    ("keep", "let's deploy to production after the code review",
             "let's deploy to production after the code review"),
    ("keep", "she parked the car in the garage",            "she parked the car in the garage"),

    # --- should NOT ANSWER (it's a transcript, not a prompt) ---------------------
    ("ask",  "what time is the standup meeting tomorrow",   "what time is the standup meeting tomorrow"),
    ("ask",  "summarize this in one sentence",              "summarize this in one sentence"),
    ("ask",  "what's the capital of France",                "what's the capital of France"),
]

import os, re, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # so `import flow` works
import flow


def norm(s: str) -> str:
    """Lowercase, collapse spaces, drop edge punctuation — compares WORDS, not style."""
    return re.sub(r"\s+", " ", s.lower()).strip(" .,!?;:\"'")


def main() -> None:
    cfg = dict(flow.load_config())
    cfg["cleanup"] = "local"                    # force on-device repair, whatever config says
    if os.environ.get("REPAIR_MODEL"):
        cfg["local_repair_model"] = os.environ["REPAIR_MODEL"]
    model = cfg.get("local_repair_model", flow.DEFAULT_CONFIG["local_repair_model"])
    print(f"model: {model}")
    print("loading (first ever run downloads ~0.9 GB)...\n")
    flow.clean("warm up the model", cfg)        # load once so per-case timings are clean

    # Ad-hoc mode: repair a single phrase from the command line.
    if len(sys.argv) > 1:
        text = " ".join(sys.argv[1:])
        print(f"in : {text}")
        print(f"out: {flow.clean(text, cfg)}")
        return

    # Suite mode.
    score = {}
    for kind, heard, want in CASES:
        t = time.time()
        out = flow.clean(heard, cfg)
        dt = (time.time() - t) * 1000
        ok = norm(out) == norm(want)
        s = score.setdefault(kind, [0, 0])
        s[0] += int(ok); s[1] += 1
        print(f"[{'✓ PASS' if ok else '✗ FAIL'}] {kind:4} ({dt:4.0f} ms)  {heard}")
        print(f"          -> {out}")
        if not ok:
            print(f"          want: {want}")
    print("\n  " + "   ".join(f"{k}: {v[0]}/{v[1]}" for k, v in score.items()))
    tot_ok = sum(v[0] for v in score.values())
    tot = sum(v[1] for v in score.values())
    print(f"  TOTAL: {tot_ok}/{tot}")


if __name__ == "__main__":
    main()
