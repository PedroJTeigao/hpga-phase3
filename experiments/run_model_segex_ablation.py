"""Model-heterogeneity follow-up: is the modal crossover cut (40 for every model in step 1) a property of
the prompt or of the model?  The P3 sec. 9.3 ablation, repeated on llama3.2:3b, qwen2.5:7b, mistral:7b.

Per model, two settings, each a fresh run of the EXISTING probe (probe_sequence_operator_compliance.py,
unchanged) restricted to crossover/segment:
  current         --segment-example current         the prompt as shipped (worked example 0-40:1, 40-100:2 and the
                                                    prose sentence "a boundary at 40 falls 40% of the way ...")
  absent_prose25  --segment-example absent_prose25  worked example removed (prompt and retry hint), prose sentence
                                                    rewritten with 25 instead of 40, so no 40 remains in the prompt
                                                    template or retry hint
100 calls, genome length 63, bounds [30, 80], temperature 0.7, seed 0, num_ctx 4096 -- the step-1 settings.
Both settings start the probe's RNG afresh from seed 0, so (while no call falls back) they see the same 100
parent pairs and the prompt text is the only difference between the two settings of a model.

The model is unloaded/loaded once per model (warm-up request) before its first setting, so the first call is not
a cold start.  Outputs (results/raw): sequence_operator_compliance_<tag>_len63_hetero_segex_<setting>.json,
llm_operator_calls_hetero_segex_<setting>_<tag>.jsonl, model_segex_<setting>_<tag>_console.log,
model_segex_driver.log.
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_model_heterogeneity_probe as h  # noqa: E402  (api/unload/warm helpers; same host and num_ctx)

MODELS = ["llama3.2:3b", "qwen2.5:7b", "mistral:7b"]
SETTINGS = ["current", "absent_prose25"]


def run_setting(model: str, setting: str) -> int:
    tag = model.replace(":", "_")
    env = dict(os.environ, HPGA_LLM_MODEL=model, HPGA_RUN_ID=f"hetero_segex_{setting}_{tag}",
               HPGA_LLM_NUM_CTX=str(h.NUM_CTX))
    env.pop("HPGA_LLM_LOG_PATH", None)
    cmd = [h.PY, str(h.PROBE), "--conditions", "crossover/segment", "--n-calls", "100", "--seed", "0",
           "--segment-example", setting, "--out-suffix", "_hetero"]
    with open(h.RAW / f"model_segex_{setting}_{tag}_console.log", "w") as f:
        return subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT, cwd=h.ROOT).returncode


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(MODELS))
    args = ap.parse_args()
    for m in args.models.split(","):
        h.unload_all()
        w = h.warm(m)
        print(f"=== {m}: warm-up load {w['load_duration_s']:.1f}s", flush=True)
        for s in SETTINGS:
            t0 = time.time()
            rc = run_setting(m, s)
            print(f"    {s}: rc={rc} wall={time.time() - t0:.0f}s", flush=True)
    h.unload_all()
    print("done", flush=True)


if __name__ == "__main__":
    main()
