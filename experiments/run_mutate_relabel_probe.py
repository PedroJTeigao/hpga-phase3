"""Model-heterogeneity step 5: separate 'round numbers' from 'early indices' in the mutate/position bias by
RELABELLING the position index in the prompt without changing the genome.

Settings (gemma4:12b and qwen2.5:7b, 100 mutate/position calls each, genome length 63, temperature 0.7, seed 0,
num_ctx 4096 -- the step-4 settings):
  normal     positions labelled 0-62, as shipped (the existing prompt and parser, untouched)
  offset100  positions labelled 100-162: the prompt states "positions labelled 100-162, the first letter is position
             100", the format line is POSITION: <100-162>, and the reply is mapped back (label - 100) before it is
             applied.  A reply outside 100-162 (e.g. an unshifted 10) is REJECTED as out of range, exactly as an
             out-of-range 0-62 label would be; it costs a retry and, after retries, a fallback.  Every raw reply, valid
             or not, is in the call log, so out-of-range answers are visible.
  offset107  (an addition beyond the two requested settings) positions labelled 107-169.  With offset 100, label 110
             is both a round number and the same index (10) as before, so a mode at 110 cannot say which; with 107 they
             separate: round labels would be 110/120, the same index would be 117, the start of the range 107.

The relabelling is done from THIS script by wrapping hpga.sequence_model._mutate_position_prompt and
_extract_position_mutation for this process; hpga/ and the probe file are not edited.  Everything else (retry hint,
system prompt, sampling, seeds, RNG stream) is unchanged; 'normal' is byte-for-byte the step-4 prompt.

  python experiments/run_mutate_relabel_probe.py              # outer: both models, all settings, one subprocess each
  python experiments/run_mutate_relabel_probe.py --inner S    # one setting in this process (model from HPGA_LLM_MODEL)

Outputs (results/raw): sequence_operator_compliance_<tag>_len63_relabel_<setting>.json,
llm_operator_calls_relabel_<setting>_<tag>.jsonl, model_relabel_<setting>_<tag>_console.log, model_relabel_driver.log.
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RAW = ROOT / "results" / "raw"
MODELS = ["gemma4:12b", "qwen2.5:7b"]
OFFSETS = {"normal": 0, "offset100": 100, "offset107": 107}


def inner(setting: str, n_calls: int, out_dir: str | None) -> None:
    sys.path.insert(0, str(HERE))
    import probe_sequence_operator_compliance as probe

    off = OFFSETS[setting]
    if off:
        sm = probe.sm

        def prompt(genome_str: str, length: int, k: int, retry_hint: str) -> str:
            lo, hi = off, off + length - 1
            return f"""Sequence (positions labelled {lo}-{hi}, the first letter is position {lo}): {genome_str}

Change exactly {k} position(s) in this sequence. For each one, name the
position and the new letter it becomes -- the new letter MUST differ from
whatever letter is currently at that position (copying the current letter
back is not a change). Do not restate the sequence.

Respond with EXACTLY {k} line(s) and nothing else, one change per line:
POSITION: <{lo}-{hi}>, NEW: <one letter from {sm._ALPHA_SET}>
{retry_hint}"""

        def extract(text: str, base: str, k: int):
            pairs = sm._POSITION_PAIR.findall(text)
            if len(pairs) != k:
                return None
            result, seen = list(base), set()
            for pos_str, letter in pairs:
                pos = int(pos_str) - off  # map the label back to the 0-based index
                if pos < 0 or pos >= len(base) or pos in seen:  # includes a label outside off..off+len-1
                    return None
                letter = letter.upper()
                if letter == base[pos]:
                    return None
                seen.add(pos)
                result[pos] = letter
            return "".join(result)

        sm._mutate_position_prompt = prompt
        sm._extract_position_mutation = extract
    argv = ["--conditions", "mutate/position", "--n-calls", str(n_calls), "--seed", "0", "--genome-len", "63",
            "--out-suffix", f"_relabel_{setting}"]
    if out_dir:
        argv += ["--out-dir", out_dir]
    probe.main(argv)


def outer() -> None:
    sys.path.insert(0, str(HERE))
    import run_model_heterogeneity_probe as h

    for m in MODELS:
        tag = m.replace(":", "_")
        h.unload_all()
        w = h.warm(m)
        print(f"=== {m}: warm-up load {w['load_duration_s']:.1f}s", flush=True)
        for s in OFFSETS:
            env = dict(os.environ, HPGA_LLM_MODEL=m, HPGA_RUN_ID=f"relabel_{s}_{tag}", HPGA_LLM_NUM_CTX=str(h.NUM_CTX))
            env.pop("HPGA_LLM_LOG_PATH", None)
            t0 = time.time()
            with open(RAW / f"model_relabel_{s}_{tag}_console.log", "w") as f:
                rc = subprocess.run([h.PY, str(Path(__file__)), "--inner", s], env=env, stdout=f,
                                    stderr=subprocess.STDOUT, cwd=ROOT).returncode
            print(f"    {s}: rc={rc} wall={time.time() - t0:.0f}s", flush=True)
    h.unload_all()
    print("done", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--inner", choices=list(OFFSETS), default=None)
    ap.add_argument("--n-calls", type=int, default=100)
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args()
    if a.inner:
        inner(a.inner, a.n_calls, a.out_dir)
    else:
        outer()
