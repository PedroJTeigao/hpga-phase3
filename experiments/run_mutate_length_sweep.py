"""Model-heterogeneity step 4a: does the modal mutate/position choice stay at the same ABSOLUTE position when the
genome length changes (a 'round-number' / salient-index effect) or does it scale with length (a RELATIVE-position
effect)?

For gemma4:12b and qwen2.5:7b, run the EXISTING probe (probe_sequence_operator_compliance.py, unchanged) on
mutate/position only, 100 calls, at genome lengths 40, 63 and 80 (the operator bounds are [30, 80]), temperature 0.7,
seed 0, num_ctx 4096.  The probe sets the mutation rate to 1/length so k = 1 at every length, keeping one independent
(position, letter) observation per call.  Each (model, length) is a fresh run, RNG restarted from seed 0.

Outputs (results/raw): sequence_operator_compliance_<tag>_len<N>_lensweep.json,
llm_operator_calls_lensweep_len<N>_<tag>.jsonl, model_lensweep_len<N>_<tag>_console.log, model_lensweep_driver.log.
"""

import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_model_heterogeneity_probe as h  # noqa: E402

MODELS = ["gemma4:12b", "qwen2.5:7b"]
LENGTHS = [40, 63, 80]


def main() -> None:
    for m in MODELS:
        tag = m.replace(":", "_")
        h.unload_all()
        w = h.warm(m)
        print(f"=== {m}: warm-up load {w['load_duration_s']:.1f}s", flush=True)
        for n in LENGTHS:
            env = dict(os.environ, HPGA_LLM_MODEL=m, HPGA_RUN_ID=f"lensweep_len{n}_{tag}", HPGA_LLM_NUM_CTX=str(h.NUM_CTX))
            env.pop("HPGA_LLM_LOG_PATH", None)
            cmd = [h.PY, str(h.PROBE), "--conditions", "mutate/position", "--n-calls", "100", "--seed", "0",
                   "--genome-len", str(n), "--out-suffix", "_lensweep"]
            t0 = time.time()
            with open(h.RAW / f"model_lensweep_len{n}_{tag}_console.log", "w") as f:
                rc = subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT, cwd=h.ROOT).returncode
            print(f"    len {n}: rc={rc} wall={time.time() - t0:.0f}s", flush=True)
    h.unload_all()
    print("done", flush=True)


if __name__ == "__main__":
    main()
