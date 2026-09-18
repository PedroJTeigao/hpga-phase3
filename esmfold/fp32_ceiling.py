"""fp32 sequence-length ceiling test: 50 residues already measured (13,801MB
peak, fits in 15GB with ~1.5GB headroom). This checks 100/150/250 in fp32,
stopping at the first OOM -- if fp32 fails by 150, that's the ceiling and
mixed precision (autocast) becomes a requirement, not an optimization; if it
holds to 250, mixed precision is optional speed/memory headroom only.
"""

import gc
import json
import random
import subprocess
import time

import torch
from transformers import AutoTokenizer, EsmForProteinFolding

AA = "ACDEFGHIKLMNPQRSTVWY"


def random_sequence(length: int, seed: int) -> str:
    rng = random.Random(seed)
    return "".join(rng.choice(AA) for _ in range(length))


def nvidia_smi_used_mb() -> float:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=10,
    ).stdout.strip()
    return float(out)


def main():
    results = {}
    tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")

    print("Loading model in fp32...", flush=True)
    t0 = time.time()
    model = EsmForProteinFolding.from_pretrained(
        "facebook/esmfold_v1", low_cpu_mem_usage=True, torch_dtype=torch.float32,
    )
    model = model.cuda()
    model.eval()
    print(f"  loaded in {time.time()-t0:.1f}s, GPU now: {nvidia_smi_used_mb():.0f}MB", flush=True)

    for length in (100, 150, 250):
        seq = random_sequence(length, seed=length)
        print(f"\n=== fp32 @ {length} residues ===", flush=True)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        inputs = tokenizer([seq], return_tensors="pt", add_special_tokens=False)
        input_ids = inputs["input_ids"].cuda()
        try:
            t0 = time.time()
            with torch.no_grad():
                output = model(input_ids)
            torch.cuda.synchronize()
            elapsed = time.time() - t0
            peak_torch_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
            peak_nvidia_mb = nvidia_smi_used_mb()
            positions = output.positions[-1, 0].detach().float().cpu()
            n_nan = int(torch.isnan(positions).sum())
            print(f"  elapsed={elapsed:.2f}s  peak_torch={peak_torch_mb:.0f}MB  "
                  f"peak_nvidia={peak_nvidia_mb:.0f}MB  positions_n_nan={n_nan}/{positions.numel()}  "
                  f"plddt_mean={float(output.plddt[0].mean()):.4f}", flush=True)
            results[str(length)] = {
                "elapsed_s": elapsed, "peak_torch_mb": peak_torch_mb,
                "peak_nvidia_mb": peak_nvidia_mb, "positions_n_nan": n_nan,
                "positions_total": positions.numel(),
            }
        except torch.cuda.OutOfMemoryError as exc:
            print(f"  OOM at {length} residues: {exc}", flush=True)
            results[str(length)] = {"error": "OOM", "detail": str(exc)}
            torch.cuda.empty_cache()
            print(f"  Stopping here per plan -- no point trying longer lengths.", flush=True)
            break

    with open("/scratch/pcanaste/projeto/phase2/esmfold/fp32_ceiling_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
