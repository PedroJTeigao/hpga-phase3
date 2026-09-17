"""Diagnostic, not part of the Phase 2 harness: measure whether concurrent
requests to the local Ollama daemon actually parallelise, or just queue
behind a single serialised worker.

Fires K concurrent /api/generate calls (short prompt, small num_predict) via
a thread pool and reports wall time and observed throughput vs K. If
throughput is flat as K increases, Ollama is serialising requests and
"parallel operator calls" would just measure its queue, not our
architecture. Must be run before any Phase 2 timing measurement.
"""

import json
import os
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ollama import Client

MODEL = os.environ.get("HPGA_PROBE_MODEL", "gemma4:12b")
HOST = "http://localhost:11434"
PROMPT = "Reply with exactly one word: OK"
NUM_PREDICT = 16
K_VALUES = [1, 2, 4, 8]
REPEATS_PER_K = 3
REQUEST_TIMEOUT_S = 300.0


def one_call(client: Client) -> float:
    t0 = time.perf_counter()
    client.generate(
        model=MODEL,
        prompt=PROMPT,
        stream=False,
        think=False,
        options={"num_predict": NUM_PREDICT, "temperature": 0.0, "seed": 0},
        keep_alive="15m",
    )
    return time.perf_counter() - t0


def run_k(k: int) -> dict:
    client = Client(host=HOST, timeout=REQUEST_TIMEOUT_S)
    # warm the model in first, untimed call so load latency doesn't pollute K=1
    client.generate(model=MODEL, prompt=PROMPT, stream=False, think=False,
                     options={"num_predict": 1}, keep_alive="15m")

    batch_wall_times = []
    per_call_latencies = []
    for _ in range(REPEATS_PER_K):
        clients = [Client(host=HOST, timeout=REQUEST_TIMEOUT_S) for _ in range(k)]
        t_batch0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=k) as pool:
            latencies = list(pool.map(one_call, clients))
        batch_wall = time.perf_counter() - t_batch0
        batch_wall_times.append(batch_wall)
        per_call_latencies.extend(latencies)

    mean_batch_wall = statistics.mean(batch_wall_times)
    throughput = k / mean_batch_wall  # requests/sec, this batch of k
    return {
        "k": k,
        "mean_batch_wall_s": mean_batch_wall,
        "throughput_req_per_s": throughput,
        "mean_per_call_latency_s": statistics.mean(per_call_latencies),
        "per_call_latencies_s": per_call_latencies,
    }


def main() -> None:
    print(f"model={MODEL} host={HOST} num_predict={NUM_PREDICT} repeats_per_k={REPEATS_PER_K}\n")
    results = []
    for k in K_VALUES:
        r = run_k(k)
        results.append(r)
        print(f"K={k:>2}  mean_batch_wall={r['mean_batch_wall_s']:.3f}s  "
              f"throughput={r['throughput_req_per_s']:.3f} req/s  "
              f"mean_per_call_latency={r['mean_per_call_latency_s']:.3f}s")

    base_throughput = results[0]["throughput_req_per_s"]
    print("\nthroughput relative to K=1 (1.0 = perfectly flat = fully serialised):")
    for r in results:
        r["relative_throughput"] = r["throughput_req_per_s"] / base_throughput
        print(f"  K={r['k']:>2}  relative_throughput={r['relative_throughput']:.3f}")

    tag = MODEL.replace(":", "_")
    out_path = (
        Path(__file__).resolve().parent.parent / "results" / "raw" / f"concurrency_probe_{tag}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"model": MODEL, "num_predict": NUM_PREDICT, "results": results}, f, indent=2)
    print(f"\nwrote results to {out_path}")


if __name__ == "__main__":
    main()
