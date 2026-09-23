"""Model-heterogeneity step 6: does the crossover cut follow the prose number across the whole range?  Steps 2-3 tried
only 40 (as shipped) and 25.  Here the prose number N is varied over 10, 25, 37, 60, 90 with the worked example removed
in every setting, so the prose number is the only thing that changes.  No GA, no fitness.

  N            prose sentence says                                  worked example
  10/25/37/60/90   "a boundary at N falls N% of the way along Parent 1 and N% of the way along Parent 2"   removed (prompt AND retry hint)

Models: qwen2.5:7b (0% fallback in steps 1-3) and gemma4:12b (ignored 25 in step 2 and went to the midpoint, 50).
100 crossover/segment calls per model per N, genome length 63, bounds [30, 80], temperature 0.7, seed 0, num_ctx 4096 --
the step-1..3 settings.  Every setting restarts the probe's RNG from seed 0, so (while no call falls back) all settings of
a model see the same 100 parent pairs and the prose number is the only difference.  N=25 is the probe's own
'absent_prose25' by construction (checked offline with --check-equivalence); it is re-run here so all five values come from
one session.

The existing probe (probe_sequence_operator_compliance.py) and hpga/ are NOT edited: this script imports the probe, adds
the settings prose10..prose90 by wrapping its apply_segment_example / unvaried_anchors for this process, and calls its
main().

Outputs (results/raw): sequence_operator_compliance_<tag>_len63_cutnum_segex_prose<N>.json,
llm_operator_calls_cutnum_prose<N>_<tag>.jsonl, model_cutnum_prose<N>_<tag>_console.log, model_cutnum_driver.log.

  python experiments/run_cut_number_sweep.py                      # outer: both models, step-6 values, one subprocess each
  python experiments/run_cut_number_sweep.py --numbers 33,45,70,80  # outer: step-7 values (same everything else)
  python experiments/run_cut_number_sweep.py --inner prose37      # one setting in this process (model from HPGA_LLM_MODEL)
  python experiments/run_cut_number_sweep.py --check-equivalence  # offline: prose25 prompt == probe's absent_prose25 prompt
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
MODELS = ["qwen2.5:7b", "gemma4:12b"]
NUMBERS = [10, 25, 37, 60, 90]  # step 6 (the default for the outer run)
STEP7_NUMBERS = [33, 45, 70, 80]  # step 7: fill the gaps; 33 replaces 30, which is the length bound printed in the prompt
SETTINGS = [f"prose{n}" for n in sorted(NUMBERS + STEP7_NUMBERS)]


def install(probe, n: int) -> None:
    """Patch the segment prompt for this process: example removed everywhere, prose number 40 -> n."""
    sm = probe.sm
    orig_prompt = sm._crossover_segment_prompt
    sm._RETRY_HINT_SEGMENT = probe._swap_example(sm._RETRY_HINT_SEGMENT, "absent", "retry hint")
    prose_n = probe._PROSE_TEXT.replace("40", str(n))

    def varied(p1_str, p2_str, n1, n2, retry_hint):
        text = probe._swap_example(orig_prompt(p1_str, p2_str, n1, n2, retry_hint), "absent", "segment prompt")
        if text.count(probe._PROSE_TEXT) != 1:
            raise SystemExit(f"expected exactly one prose sentence, found {text.count(probe._PROSE_TEXT)}")
        return text.replace(probe._PROSE_TEXT, prose_n)

    sm._crossover_segment_prompt = varied


def inner(setting: str, n_calls: int, out_dir: str | None) -> None:
    sys.path.insert(0, str(HERE))
    import probe_sequence_operator_compliance as probe

    n = int(setting.removeprefix("prose"))
    probe.SEGMENT_EXAMPLE_SETTINGS = probe.SEGMENT_EXAMPLE_SETTINGS + (setting,)
    orig_apply, orig_anchors = probe.apply_segment_example, probe.unvaried_anchors

    def apply_segment_example(s: str) -> None:
        if s == setting:
            install(probe, n)
        else:
            orig_apply(s)

    def unvaried_anchors(s: str) -> list[str]:
        if s != setting:
            return orig_anchors(s)
        return [f"prose sentence changed to use {n} instead of 40 ({n} is the only illustrative number)",
                "worked example removed from the prompt and the retry hint", probe._TEMPLATE_ANCHOR]

    probe.apply_segment_example, probe.unvaried_anchors = apply_segment_example, unvaried_anchors
    argv = ["--conditions", "crossover/segment", "--n-calls", str(n_calls), "--seed", "0",
            "--segment-example", setting, "--out-suffix", "_cutnum"]
    if out_dir:
        argv += ["--out-dir", out_dir]
    probe.main(argv)


def check_equivalence() -> None:
    """Offline: for a fixed parent pair, the prose25 prompt (and retry hint) equals the probe's absent_prose25 one."""
    os.environ.setdefault("HPGA_LLM_MODEL", "qwen2.5:7b")
    sys.path.insert(0, str(HERE))
    import random
    import probe_sequence_operator_compliance as probe

    sm = probe.sm
    orig = sm._crossover_segment_prompt, sm._RETRY_HINT_SEGMENT
    rng = random.Random(0)
    p1, p2 = sm.random_sequence(rng), sm.random_sequence(rng)
    args = lambda: ("".join(p1), "".join(p2), len(p1), len(p2))  # noqa: E731

    probe.apply_segment_example("absent_prose25")
    ref = sm._crossover_segment_prompt(*args(), sm._RETRY_HINT_SEGMENT), sm._RETRY_HINT_SEGMENT
    sm._crossover_segment_prompt, sm._RETRY_HINT_SEGMENT = orig
    install(probe, 25)
    new = sm._crossover_segment_prompt(*args(), sm._RETRY_HINT_SEGMENT), sm._RETRY_HINT_SEGMENT
    print("prompt identical:", ref[0] == new[0], "| retry hint identical:", ref[1] == new[1])
    if ref != new:
        raise SystemExit("prose25 differs from absent_prose25")
    sm._crossover_segment_prompt, sm._RETRY_HINT_SEGMENT = orig
    for n in NUMBERS + STEP7_NUMBERS:
        install(probe, n)
        text = sm._crossover_segment_prompt(*args(), sm._RETRY_HINT_SEGMENT)
        print(f"N={n}: 'boundary at {n} falls {n}%' x{text.count(f'boundary at {n} falls {n}%')}, 'e.g.' x{text.count('e.g.')}, "
              f"retry hint 'e.g.' x{sm._RETRY_HINT_SEGMENT.count('e.g.')}")
        sm._crossover_segment_prompt, sm._RETRY_HINT_SEGMENT = orig


def outer(numbers: list[int]) -> None:
    sys.path.insert(0, str(HERE))
    import run_model_heterogeneity_probe as h

    for m in MODELS:
        tag = m.replace(":", "_")
        h.unload_all()
        w = h.warm(m)
        print(f"=== {m}: warm-up load {w['load_duration_s']:.1f}s", flush=True)
        for s in [f"prose{n}" for n in numbers]:
            env = dict(os.environ, HPGA_LLM_MODEL=m, HPGA_RUN_ID=f"cutnum_{s}_{tag}", HPGA_LLM_NUM_CTX=str(h.NUM_CTX))
            env.pop("HPGA_LLM_LOG_PATH", None)
            t0 = time.time()
            with open(RAW / f"model_cutnum_{s}_{tag}_console.log", "w") as f:
                rc = subprocess.run([h.PY, str(Path(__file__)), "--inner", s], env=env, stdout=f,
                                    stderr=subprocess.STDOUT, cwd=ROOT).returncode
            print(f"    {s}: rc={rc} wall={time.time() - t0:.0f}s", flush=True)
    h.unload_all()
    print("done", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--inner", choices=SETTINGS, default=None)
    ap.add_argument("--numbers", default=",".join(map(str, NUMBERS)), help="outer run: comma-separated prose numbers")
    ap.add_argument("--n-calls", type=int, default=100)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--check-equivalence", action="store_true")
    a = ap.parse_args()
    if a.check_equivalence:
        check_equivalence()
    elif a.inner:
        inner(a.inner, a.n_calls, a.out_dir)
    else:
        outer([int(x) for x in a.numbers.split(",")])
