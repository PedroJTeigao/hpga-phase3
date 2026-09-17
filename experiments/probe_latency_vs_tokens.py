"""Reconcile the concurrency probe's ~3.3s/call (trivial prompt, ~20 tokens
in) against the pilot's 11.4s/call (~190 tokens in) by measuring latency as
a function of tokens_in and tokens_out directly, holding the other roughly
fixed in each sweep. Fits latency_s ~= a + b*tokens_in + c*tokens_out by
least squares so the per-token constants are explicit, not eyeballed.

Sweep A holds output short (num_predict=16) and pads the prompt with inert
filler text to vary tokens_in.
Sweep B holds input short and asks for a long, easy-to-satisfy fixed-length
output ("print digit 7 N times") to vary tokens_out.

temperature=0 here (unlike the operators' default 0.7) -- this probe is
about raw latency, not operator behaviour, and determinism keeps repeats
comparable.

Writes raw per-call data to results/raw/latency_vs_tokens.jsonl.
"""

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from ollama import Client

MODEL = os.environ.get("HPGA_PROBE_MODEL", "gemma4:12b")
HOST = "http://localhost:11434"
_tag = MODEL.replace(":", "_")
LOG_PATH = (
    Path(__file__).resolve().parent.parent / "results" / "raw" / "latency_vs_tokens.jsonl"
    if MODEL == "gemma4:12b"
    else Path(__file__).resolve().parent.parent / "results" / "raw" / f"latency_vs_tokens_{_tag}.jsonl"
)

FILLER = "context "  # short, low-entropy filler word to pad tokens_in
PAD_LEVELS = [0, 50, 150, 300, 600]
OUT_LEVELS = [8, 16, 32, 64, 128, 256]
REPEATS = 3


def call(client: Client, prompt: str, num_predict: int, seed: int):
    t0 = time.perf_counter()
    r = client.generate(
        model=MODEL, prompt=prompt, stream=False, think=False,
        options={"num_predict": num_predict, "temperature": 0.0, "seed": seed},
        keep_alive="30m",
    )
    latency = time.perf_counter() - t0
    return latency, r.prompt_eval_count or 0, r.eval_count or 0


def padded_prompt(pad_words: int) -> str:
    filler = FILLER * pad_words
    return f"{filler}\nReply with exactly one word: OK"


def repeat_prompt(n: int) -> str:
    return f"Print the digit 7 exactly {n} times, separated by single spaces, and nothing else."


def main() -> None:
    print(f"model={MODEL} host={HOST} log_path={LOG_PATH}\n")
    client = Client(host=HOST, timeout=180.0)
    call(client, "Reply with exactly one word: OK", 8, 0)  # warm, untimed

    results = []

    print("=== Sweep A: vary tokens_in, tokens_out held small ===")
    for pad in PAD_LEVELS:
        for rep in range(REPEATS):
            lat, tin, tout = call(client, padded_prompt(pad), 16, seed=rep)
            results.append({"sweep": "A", "pad_words": pad, "rep": rep,
                             "latency_s": lat, "tokens_in": tin, "tokens_out": tout})
            print(f"  pad={pad:>4} rep={rep} tokens_in={tin:>4} tokens_out={tout:>3} latency={lat:.3f}s")

    print("\n=== Sweep B: vary tokens_out, tokens_in held small ===")
    for n in OUT_LEVELS:
        for rep in range(REPEATS):
            lat, tin, tout = call(client, repeat_prompt(n), n + 16, seed=rep)
            results.append({"sweep": "B", "target_n": n, "rep": rep,
                             "latency_s": lat, "tokens_in": tin, "tokens_out": tout})
            print(f"  target_n={n:>4} rep={rep} tokens_in={tin:>4} tokens_out={tout:>3} latency={lat:.3f}s")

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"\nwrote {len(results)} calls to {LOG_PATH}")

    X = np.array([[1.0, r["tokens_in"], r["tokens_out"]] for r in results])
    y = np.array([r["latency_s"] for r in results])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    a, b, c = coef
    y_pred = X @ coef
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    print(f"\nfit: latency_s ~= {a:.4f} + {b:.5f} * tokens_in + {c:.5f} * tokens_out   (R^2={r2:.4f})")
    print(f"  fixed overhead per call: {a:.3f}s")
    print(f"  marginal cost per input token:  {b*1e3:.3f} ms")
    print(f"  marginal cost per output token: {c*1e3:.3f} ms")

    # sanity: predict the two already-observed regimes
    probe_pred = a + b * 20 + c * 4  # concurrency probe: ~20 tokens in, ~2-4 out
    pilot_pred = a + b * 190 + c * 29  # pilot: mean tokens_in=190, tokens_out=29 (5121/27, 792/27)
    print(f"\n  predicted latency at concurrency-probe scale (tokens_in~20, tokens_out~4): {probe_pred:.3f}s "
          f"(observed ~3.3s)")
    print(f"  predicted latency at pilot scale (tokens_in~190, tokens_out~29): {pilot_pred:.3f}s "
          f"(observed 11.425s mean)")


if __name__ == "__main__":
    main()
