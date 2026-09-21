"""Model-heterogeneity step 3b: separate the two things the PHASE3 sec. 9.3 / step-2 ablation changed together.

The step-2 setting 'absent_prose25' removed the worked example AND changed the prose number from 40 to 25, so it
cannot say which of the two carries the effect.  qwen2.5:7b (0/100 fallback at step 1) is used for a 2x2:

  setting              prose sentence says   worked example
  current              40                    0-40:1, 40-100:2   (as shipped)
  absent               40                    removed            (the probe's own 'absent' setting)
  prose25_example40    25                    0-40:1, 40-100:2   (new here)
  absent_prose25       25                    removed            (the probe's own setting)

'Prose sentence' = "a boundary at N falls N% of the way along Parent 1 and N% of the way along Parent 2".
The example is varied wherever it occurs (prompt and retry hint), as in the probe.  prose25_example40 leaves the
example and the retry hint's example untouched and changes only the prose sentence.

The existing probe (probe_sequence_operator_compliance.py) is NOT edited: this script imports it, adds the one new
setting by wrapping its apply_segment_example / unvaried_anchors for this process, and calls its main().  100
crossover/segment calls per setting, genome length 63, bounds [30, 80], temperature 0.7, seed 0, num_ctx 4096.

Outputs (results/raw): sequence_operator_compliance_qwen2.5_7b_len63_factorial_segex_<setting>.json,
llm_operator_calls_factorial_<setting>_qwen2.5_7b.jsonl, model_factorial_<setting>_console.log, model_factorial_driver.log.

  python experiments/run_qwen_segex_factorial.py            # outer: all four settings, one subprocess each
  python experiments/run_qwen_segex_factorial.py --inner S  # one setting in this process
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
MODEL = "qwen2.5:7b"
SETTINGS = ["current", "absent", "prose25_example40", "absent_prose25"]
NEW = "prose25_example40"


def inner(setting: str, n_calls: int, out_dir: str | None) -> None:
    os.environ["HPGA_LLM_MODEL"] = MODEL
    sys.path.insert(0, str(HERE))
    import probe_sequence_operator_compliance as probe

    probe.SEGMENT_EXAMPLE_SETTINGS = probe.SEGMENT_EXAMPLE_SETTINGS + (NEW,)
    orig_apply, orig_anchors = probe.apply_segment_example, probe.unvaried_anchors

    def apply_segment_example(s: str) -> None:
        if s != NEW:
            return orig_apply(s)
        orig_prompt = probe.sm._crossover_segment_prompt

        def varied(p1_str, p2_str, n1, n2, retry_hint):
            text = orig_prompt(p1_str, p2_str, n1, n2, retry_hint)
            if text.count(probe._PROSE_TEXT) != 1:
                raise SystemExit(f"expected exactly one prose sentence, found {text.count(probe._PROSE_TEXT)}")
            return text.replace(probe._PROSE_TEXT, probe._PROSE_TEXT_25)

        probe.sm._crossover_segment_prompt = varied

    def unvaried_anchors(s: str) -> list[str]:
        if s != NEW:
            return orig_anchors(s)
        return ["worked example 0-40:1, 40-100:2 kept in the prompt AND the retry hint (deliberately unvaried)",
                "prose sentence changed to use 25 instead of 40", probe._TEMPLATE_ANCHOR]

    probe.apply_segment_example, probe.unvaried_anchors = apply_segment_example, unvaried_anchors
    argv = ["--conditions", "crossover/segment", "--n-calls", str(n_calls), "--seed", "0",
            "--segment-example", setting, "--out-suffix", "_factorial"]
    if out_dir:
        argv += ["--out-dir", out_dir]
    probe.main(argv)


def outer() -> None:
    sys.path.insert(0, str(HERE))
    import run_model_heterogeneity_probe as h

    h.unload_all()
    w = h.warm(MODEL)
    print(f"=== {MODEL}: warm-up load {w['load_duration_s']:.1f}s", flush=True)
    for s in SETTINGS:
        env = dict(os.environ, HPGA_RUN_ID=f"factorial_{s}_qwen2.5_7b", HPGA_LLM_NUM_CTX=str(h.NUM_CTX))
        env.pop("HPGA_LLM_LOG_PATH", None)
        t0 = time.time()
        with open(RAW / f"model_factorial_{s}_console.log", "w") as f:
            rc = subprocess.run([h.PY, str(Path(__file__)), "--inner", s], env=env, stdout=f,
                                stderr=subprocess.STDOUT, cwd=ROOT).returncode
        print(f"    {s}: rc={rc} wall={time.time() - t0:.0f}s", flush=True)
    h.unload_all()
    print("done", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--inner", choices=SETTINGS, default=None)
    ap.add_argument("--n-calls", type=int, default=100)
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args()
    if a.inner:
        inner(a.inner, a.n_calls, a.out_dir)
    else:
        outer()
