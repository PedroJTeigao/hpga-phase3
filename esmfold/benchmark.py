"""ESMFold precision/throughput benchmark on this node's T4.

Stage 1: try loading + running the model in fp32 at 50 residues. The
weight-size arithmetic (3.525B params x 4 bytes = ~14.1GB) suggested this
might not fit in 15GB VRAM, but that's arithmetic on total params, not a
measurement -- the checkpoint may not hold every parameter in fp32, and
torch/accelerate can offload. Measured here rather than assumed.

Stage 2: if stage 1 succeeds, reload in fp16, run the *same* 50-residue
sequence, and diff the predicted coordinates against stage 1's output.
This is the more important comparison -- ESMFold is the fitness oracle
for the eventual search, so if fp16 shifts the predicted structure, the
search inherits a precision dependency that would otherwise show up as
unexplained noise later.

Stage 3: fp16 benchmark at 50/100/150/250 residues -- time and peak VRAM
(both torch's own accounting and nvidia-smi, since they can disagree).
"""

import gc
import json
import random
import subprocess
import sys
import time

import torch
from transformers import AutoTokenizer, EsmForProteinFolding
import transformers.models.esm.modeling_esmfold as _esmfold_mod

AA = "ACDEFGHIKLMNPQRSTVWY"  # 20 standard amino acids

# fp16 numerical-stability patch. compute_tm's argmax-via-equality
# (`weighted == torch.max(weighted)`) produces zero matches in fp16
# (IndexError: index 0 out of bounds for dimension 0 with size 0).
# Upcasting just this function's own input to fp32 (first attempt) did NOT
# fix it -- the crash reproduced identically, which means the bad value
# (almost certainly NaN) is already present in `ptm_logits` itself, i.e.
# upstream in the fp16 trunk/pair-representation (`structure["s_z"]`,
# modeling_esmfold.py line 2167), not introduced by compute_tm's own
# arithmetic. That's a more significant finding than a numeric-precision
# question: naive fp16 casting may be producing NaN/Inf internally, not
# just less-precise numbers. So instead of a "fix," this makes the pTM/PAE
# heads fail SOFT (return NaN instead of raising) so the forward pass
# completes and `output.positions` -- what the fitness oracle actually
# needs -- can be inspected directly for corruption, separately from
# whatever happened to this auxiliary confidence head.
_orig_compute_tm = _esmfold_mod.compute_tm
_orig_compute_pae = _esmfold_mod.compute_predicted_aligned_error


def _safe_compute_tm(logits, *args, **kwargs):
    try:
        return _orig_compute_tm(logits, *args, **kwargs)
    except IndexError:
        print("  [patch] compute_tm hit the empty-argmax case (logits likely "
              "contain NaN/Inf) -- returning NaN instead of crashing", flush=True)
        return torch.tensor(float("nan"), device=logits.device)


def _safe_compute_pae(logits, *args, **kwargs):
    try:
        return _orig_compute_pae(logits, *args, **kwargs)
    except (IndexError, RuntimeError) as exc:
        print(f"  [patch] compute_predicted_aligned_error failed ({exc}) -- "
              f"returning empty dict instead of crashing", flush=True)
        return {}


_esmfold_mod.compute_tm = _safe_compute_tm
_esmfold_mod.compute_predicted_aligned_error = _safe_compute_pae


def random_sequence(length: int, seed: int) -> str:
    rng = random.Random(seed)
    return "".join(rng.choice(AA) for _ in range(length))


def nvidia_smi_used_mb() -> float:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=10,
    ).stdout.strip()
    return float(out)


def load_model(dtype: torch.dtype):
    print(f"Loading model in {dtype}...", flush=True)
    t0 = time.time()
    model = EsmForProteinFolding.from_pretrained(
        "facebook/esmfold_v1", low_cpu_mem_usage=True, torch_dtype=dtype,
    )
    model = model.cuda()
    model.eval()
    print(f"  loaded + moved to GPU in {time.time()-t0:.1f}s", flush=True)
    return model


def run_one(model, tokenizer, sequence: str) -> dict:
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    inputs = tokenizer([sequence], return_tensors="pt", add_special_tokens=False)
    input_ids = inputs["input_ids"].cuda()

    t0 = time.time()
    with torch.no_grad():
        output = model(input_ids)
    torch.cuda.synchronize()
    elapsed = time.time() - t0

    peak_torch_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    peak_nvidia_mb = nvidia_smi_used_mb()

    # output.positions: [n_recycles, batch, L, 14 atoms, 3] -- final recycle,
    # CA atom (index 1 in the standard 14-atom ordering), first (only) batch.
    positions = output.positions[-1, 0].detach().float().cpu()  # [L, 14, 3]
    n_nan = int(torch.isnan(positions).sum())
    n_inf = int(torch.isinf(positions).sum())
    if n_nan or n_inf:
        print(f"  !!! positions contain {n_nan} NaN / {n_inf} Inf values out of "
              f"{positions.numel()} !!!", flush=True)

    return {
        "elapsed_s": elapsed,
        "peak_torch_mb": peak_torch_mb,
        "peak_nvidia_mb": peak_nvidia_mb,
        "positions": positions,
        "positions_n_nan": n_nan,
        "positions_n_inf": n_inf,
        "plddt_mean": float(output.plddt[0].mean()) if hasattr(output, "plddt") else None,
    }


def free_model(model):
    """`del model` inside this function only clears the local binding, not
    the caller's own reference to the same object -- caught in testing when
    stage 2's fp16 load OOM'd because stage 1's fp32 model (~13.8GB) was
    still fully resident. Returns None so call sites reassign their own
    variable (`model32 = free_model(model32)`), which actually drops the
    last reference and lets gc/empty_cache do something."""
    del model
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    return None


def main():
    results = {}
    tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
    seq50 = random_sequence(50, seed=0)
    print(f"seq50 = {seq50}", flush=True)

    # --- Stage 1: fp32 @ 50 residues ------------------------------------
    print("\n=== STAGE 1: fp32 @ 50 residues ===", flush=True)
    fp32_ok = False
    fp32_positions = None
    try:
        model32 = load_model(torch.float32)
        r = run_one(model32, tokenizer, seq50)
        fp32_ok = True
        fp32_positions = r["positions"]
        print(f"  fp32 @ 50: elapsed={r['elapsed_s']:.2f}s  "
              f"peak_torch={r['peak_torch_mb']:.0f}MB  peak_nvidia={r['peak_nvidia_mb']:.0f}MB  "
              f"plddt_mean={r['plddt_mean']}", flush=True)
        results["fp32_50"] = {k: v for k, v in r.items() if k != "positions"}
        model32 = free_model(model32)
    except torch.cuda.OutOfMemoryError as exc:
        print(f"  fp32 @ 50: OOM -- {exc}", flush=True)
        results["fp32_50"] = {"error": "OOM", "detail": str(exc)}
        torch.cuda.empty_cache()
    except Exception as exc:
        print(f"  fp32 @ 50: FAILED (not OOM) -- {type(exc).__name__}: {exc}", flush=True)
        results["fp32_50"] = {"error": type(exc).__name__, "detail": str(exc)}
        torch.cuda.empty_cache()

    print(f"\nGPU after stage 1: {nvidia_smi_used_mb():.0f}MB used", flush=True)

    # --- Stage 2: fp16 @ 50 residues, diff against fp32 if it ran -------
    print("\n=== STAGE 2: fp16 @ 50 residues ===", flush=True)
    print(f"GPU before stage 2: {nvidia_smi_used_mb():.0f}MB used", flush=True)
    model16 = load_model(torch.float16)
    r50 = run_one(model16, tokenizer, seq50)
    print(f"  fp16 @ 50: elapsed={r50['elapsed_s']:.2f}s  "
          f"peak_torch={r50['peak_torch_mb']:.0f}MB  peak_nvidia={r50['peak_nvidia_mb']:.0f}MB  "
          f"plddt_mean={r50['plddt_mean']}", flush=True)
    results["fp16_50"] = {k: v for k, v in r50.items() if k != "positions"}

    if fp32_ok:
        diff = (fp32_positions - r50["positions"]).abs()
        ca = diff[:, 1, :]  # CA atom per residue
        print(f"\n  fp32 vs fp16 coordinate diff (all 14 atoms x 3 dims, Angstrom):", flush=True)
        print(f"    max_abs_diff={diff.max().item():.4f}  mean_abs_diff={diff.mean().item():.4f}", flush=True)
        print(f"  fp32 vs fp16 CA-only diff:", flush=True)
        print(f"    max_abs_diff={ca.max().item():.4f}  mean_abs_diff={ca.mean().item():.4f}  "
              f"rmsd={(ca.pow(2).sum(-1).mean().sqrt()).item():.4f}", flush=True)
        results["fp32_vs_fp16_diff"] = {
            "all_atoms_max_abs": diff.max().item(),
            "all_atoms_mean_abs": diff.mean().item(),
            "ca_max_abs": ca.max().item(),
            "ca_mean_abs": ca.mean().item(),
            "ca_rmsd": (ca.pow(2).sum(-1).mean().sqrt()).item(),
        }
    else:
        print("  (fp32 did not run, no comparison possible)", flush=True)
        results["fp32_vs_fp16_diff"] = None

    # --- Stage 3: fp16 benchmark @ 50/100/150/250 -----------------------
    print("\n=== STAGE 3: fp16 benchmark, 50/100/150/250 residues ===", flush=True)
    results["fp16_benchmark"] = {"50": {k: v for k, v in r50.items() if k != "positions"}}
    for length in (100, 150, 250):
        seq = random_sequence(length, seed=length)
        try:
            r = run_one(model16, tokenizer, seq)
            print(f"  fp16 @ {length}: elapsed={r['elapsed_s']:.2f}s  "
                  f"peak_torch={r['peak_torch_mb']:.0f}MB  peak_nvidia={r['peak_nvidia_mb']:.0f}MB  "
                  f"plddt_mean={r['plddt_mean']}", flush=True)
            results["fp16_benchmark"][str(length)] = {k: v for k, v in r.items() if k != "positions"}
        except torch.cuda.OutOfMemoryError as exc:
            print(f"  fp16 @ {length}: OOM -- {exc}", flush=True)
            results["fp16_benchmark"][str(length)] = {"error": "OOM", "detail": str(exc)}
            torch.cuda.empty_cache()
        except Exception as exc:
            print(f"  fp16 @ {length}: FAILED (not OOM) -- {type(exc).__name__}: {exc}", flush=True)
            results["fp16_benchmark"][str(length)] = {"error": type(exc).__name__, "detail": str(exc)}
            torch.cuda.empty_cache()

    with open("/scratch/pcanaste/projeto/phase2/esmfold/benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
