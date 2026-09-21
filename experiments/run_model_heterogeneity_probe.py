"""Model-heterogeneity step 1: does the format compliance / position / cut collapse seen on
gemma4:12b hold for other models?  No GA, no fitness, no ESMFold.

For each model, in order:
  1. unload whatever this user's Ollama server has resident (the API call, keep_alive=0; never a
     restart), snapshot the GPU;
  2. one tiny warm-up request at the operators' own num_ctx (4096): Ollama's load_duration is the
     model LOAD time; /api/ps and nvidia-smi after it give the resident memory;
  3. the EXISTING probe (experiments/probe_sequence_operator_compliance.py), unchanged, as a
     subprocess: --conditions mutate/position,crossover/segment --n-calls 100 --seed 0, genome
     length 63, bounds [30, 80] (hpga.sequence_model), temperature 0.7 (operators default).
  Then every pair of the models is loaded together (tiny requests) to see whether two fit on the
  GPU at once and stay resident.

Outputs (results/raw): sequence_operator_compliance_<tag>_len63_hetero.json (the probe's own file),
llm_operator_calls_hetero_<tag>.jsonl (every request), model_heterogeneity_<tag>_console.log,
model_heterogeneity_meta.json (load time, memory, pair-fit results).
"""

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "results" / "raw"
HOST = os.environ.get("HPGA_OLLAMA_HOST", "http://localhost:11434")
NUM_CTX = 4096  # the operators' default (HPGA_LLM_NUM_CTX); footprint is measured at this setting
MODELS = ["gemma4:12b", "llama3.2:3b", "qwen2.5:7b", "mistral:7b"]  # gemma4:12b = the reference
PROBE = ROOT / "experiments" / "probe_sequence_operator_compliance.py"
PY = os.environ.get("PROBE_PYTHON", sys.executable)


def api(path: str, payload: dict | None = None, timeout: float = 900.0) -> dict:
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(HOST + path, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def gpu() -> dict:
    def run(cmd):
        try:
            return subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout.strip()
        except Exception as exc:  # pragma: no cover
            return f"<failed: {exc!r}>"
    used = run(["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"])
    apps = run(["nvidia-smi", "--query-compute-apps=pid,used_memory,process_name", "--format=csv,noheader"])
    return {"used_total_util": used, "compute_apps": apps or "(none)"}


def used_mib() -> int:
    return int(gpu()["used_total_util"].split(",")[0])


def loaded() -> list[dict]:
    return api("/api/ps").get("models", [])


def unload_all() -> list[str]:
    names = [m["name"] for m in loaded()]
    for n in names:
        api("/api/generate", {"model": n, "keep_alive": 0, "stream": False})
    for _ in range(30):  # wait until the server reports empty and the driver has released VRAM
        if not loaded():
            break
        time.sleep(1)
    time.sleep(2)
    return names


def warm(model: str) -> dict:
    t0 = time.perf_counter()
    r = api("/api/generate", {"model": model, "prompt": "Reply with the single word OK.", "stream": False,
                              "options": {"num_ctx": NUM_CTX, "num_predict": 4, "temperature": 0},
                              "keep_alive": "30m"})
    return {"wall_s": time.perf_counter() - t0, "load_duration_s": r.get("load_duration", 0) / 1e9,
            "total_duration_s": r.get("total_duration", 0) / 1e9, "prompt_eval_count": r.get("prompt_eval_count"),
            "eval_count": r.get("eval_count")}


def ps_entry(model: str) -> dict | None:
    for m in loaded():
        if m["name"] == model:
            return {"size_bytes": m.get("size"), "size_vram_bytes": m.get("size_vram"),
                    "context_length": m.get("context_length"),
                    "fully_on_gpu": m.get("size") == m.get("size_vram")}
    return None


def show(model: str) -> dict:
    d = api("/api/show", {"model": model})
    det = d.get("details", {})
    tag = next((m for m in api("/api/tags")["models"] if m["name"] == model), {})
    return {"family": det.get("family"), "parameter_size": det.get("parameter_size"),
            "quantization": det.get("quantization_level"), "disk_bytes": tag.get("size"), "digest": tag.get("digest")}


def measure(model: str) -> dict:
    was_loaded = unload_all()
    g0 = gpu()
    base = used_mib()
    w = warm(model)
    time.sleep(2)
    g1 = gpu()
    return {"model": model, "info": show(model), "unloaded_before": was_loaded, "gpu_before": g0,
            "warmup": w, "ps": ps_entry(model), "gpu_after_load": g1,
            "gpu_used_mib_delta": used_mib() - base}


def run_probe(model: str) -> int:
    tag = model.replace(":", "_")
    env = dict(os.environ, HPGA_LLM_MODEL=model, HPGA_RUN_ID=f"hetero_{tag}", HPGA_LLM_NUM_CTX=str(NUM_CTX))
    env.pop("HPGA_LLM_LOG_PATH", None)
    log = RAW / f"model_heterogeneity_{tag}_console.log"
    cmd = [PY, str(PROBE), "--conditions", "mutate/position,crossover/segment", "--n-calls", "100",
           "--seed", "0", "--out-suffix", "_hetero"]
    with open(log, "w") as f:
        return subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT, cwd=ROOT).returncode


def pair_fit(a: str, b: str) -> dict:
    unload_all()
    base = used_mib()
    warm(a)
    ps_a = ps_entry(a)
    warm(b)
    both = {m["name"]: {"size_bytes": m.get("size"), "size_vram_bytes": m.get("size_vram")} for m in loaded()}
    g = gpu()
    used = used_mib() - base
    # touch the first model again: if the server evicted it to make room, this reloads it
    t = warm(a)
    after = sorted(m["name"] for m in loaded())
    return {"pair": [a, b], "resident_after_both": sorted(both), "both_resident": a in both and b in both,
            "both_fully_on_gpu": all(v["size_bytes"] == v["size_vram_bytes"] for v in both.values()) and len(both) == 2,
            "sizes": both, "gpu_used_mib_delta": used, "gpu": g,
            "retouch_first_load_s": t["load_duration_s"], "resident_after_retouch": after,
            "first_model_evicted": a not in both or t["load_duration_s"] > 1.0}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--skip-probe", action="store_true")
    ap.add_argument("--skip-pairs", action="store_true")
    args = ap.parse_args()
    models = args.models.split(",")
    meta_path = RAW / "model_heterogeneity_meta.json"
    meta = json.load(open(meta_path)) if meta_path.exists() else {"models": {}, "pairs": []}
    meta["config"] = {"num_ctx": NUM_CTX, "n_calls_per_operator": 100, "seed": 0, "genome_len": 63,
                      "temperature": 0.7, "host": HOST, "server_env_note": "OLLAMA_CONTEXT_LENGTH=8192, "
                      "OLLAMA_KEEP_ALIVE=-1 on the server; requests set num_ctx=4096, keep_alive=30m"}
    for m in models:
        print(f"=== {m}: load/memory measurement", flush=True)
        entry = measure(m)
        meta["models"][m] = entry
        json.dump(meta, open(meta_path, "w"), indent=2)
        print(f"    load_duration={entry['warmup']['load_duration_s']:.1f}s  ps={entry['ps']}  "
              f"gpu_delta={entry['gpu_used_mib_delta']} MiB", flush=True)
        if not args.skip_probe:
            t0 = time.time()
            rc = run_probe(m)
            entry["probe"] = {"returncode": rc, "wall_s": time.time() - t0}
            json.dump(meta, open(meta_path, "w"), indent=2)
            print(f"    probe rc={rc} wall={time.time() - t0:.0f}s", flush=True)
    if not args.skip_pairs:
        meta["pairs"] = []
        for a, b in combinations(models, 2):
            print(f"=== pair {a} + {b}", flush=True)
            r = pair_fit(a, b)
            meta["pairs"].append(r)
            json.dump(meta, open(meta_path, "w"), indent=2)
            print(f"    both_resident={r['both_resident']} fully_on_gpu={r['both_fully_on_gpu']} "
                  f"delta={r['gpu_used_mib_delta']} MiB retouch_load={r['retouch_first_load_s']:.1f}s", flush=True)
        unload_all()
    print("done", flush=True)


if __name__ == "__main__":
    main()
