# ESMFold on ece000: Setup, Precision, and Throughput

*T4 (15360 MiB, compute capability 7.5), no root, no scheduler, shared node.
`torch 2.11.0+cu128` / `transformers 5.17.0` pre-existing in
`/scratch/pcanaste/venv` (predates this work by 3 days — not newly
installed). Scripts and logs: `/scratch/pcanaste/projeto/esmfold/`.*

## 1. Install and weights

No new packages needed — `EsmForProteinFolding`/`AutoTokenizer` imported
cleanly from the existing `transformers` install. Cache redirects (none
touching `$HOME`):

| var | value | status |
|---|---|---|
| `HF_HOME` | `/scratch/pcanaste/cache/huggingface` | already set, persisted in `~/.bashrc` |
| `XDG_CACHE_HOME` | `/scratch/pcanaste/cache` | already set, persisted |
| `TORCH_HOME` | `/scratch/pcanaste/cache/torch` | added, persisted this session |

**Weights**: `facebook/esmfold_v1`, one 7.9GB blob at
`/scratch/pcanaste/cache/huggingface/hub/models--facebook--esmfold_v1/`.
3,525,038,915 parameters — confirms the checkpoint bundles the full
ESM-2 3B-parameter language model as its sequence backbone, not a smaller
distilled variant. `contact_head.regression.{weight,bias}` report MISSING
on load — benign, an auxiliary contact-prediction head not included in
the folding checkpoint.

## 2. The AFS backup failed silently the first time — worth its own note

Per plan, the 7.9GB blob was copied to `/afs/ece.cmu.edu/usr/pcanaste/models/`
(16% of the 50GB budget) so it survives the 28-day `/scratch` wipe without
re-downloading. The first copy **appeared to succeed** — `cp` exited 0, no
error printed — but the AFS Kerberos token expired mid-transfer. The
result: a **4.13GB file (of 7.9GB expected)**, silently truncated, owned by
an unauthenticated identity (`uid 990838480`, not `pcanaste`) that this
session couldn't even read back afterward (`tokens` showed none held).

Caught only because the copy was checked, not because anything reported
failure: `ls -la` on the destination showed the wrong size, and `tokens`
showed the Cache Manager holding nothing. Fixed by `aklog` (Kerberos
ticket was still valid, just hadn't been converted to an AFS token —
`aklog -cell ece.cmu.edu` specifically, since the plain `aklog` only
covered `andrew.cmu.edu`), then deleting the truncated file and redoing
the copy. Verified this time with an actual checksum, not just a clean
exit code:

```
16e8381bfb8f8eefa7897f656d543255  (source, /scratch)
16e8381bfb8f8eefa7897f656d543255  (AFS copy)
```

**This is the same class of failure flagged earlier in the HPGA project's
own history** (an untracked results directory living on a wipeable disk,
discovered only because it was checked, not because anything alerted on
it) — an operation that reports success and didn't actually do the thing.
The concrete change this session makes going forward: **AFS writes get a
checksum comparison as a default step, not a fix applied after being
caught.** A clean exit code from `cp` across AFS is not evidence of a
complete, correctly-owned file.

## 3. fp32: works across the full requested range, headroom shrinking

| length | elapsed | peak (torch) | peak (nvidia-smi) | headroom vs. 15360MB |
|---|---|---|---|---|
| 50 | 2.2-2.4s | 13,638 MB | 13,801 MB | 1,559 MB |
| 100 | 2.91s | 13,684 MB | 13,889 MB | 1,471 MB |
| 150 | 5.15s | 13,839 MB | 14,049 MB | 1,311 MB |
| 250 | 16.09s | 14,423 MB | 14,749 MB | 611 MB |

No OOM at any tested length — the naive weight-size arithmetic (3.525B
params x 4 bytes = 14.1GB) that suggested fp32 might not fit was measured
wrong, not just imprecise: actual peak usage at 50 residues (13.8GB) is
below that estimate, and stays below the card's 15GB ceiling through 250
residues. **fp32 alone covers the full requested length range on this
hardware.** Headroom does shrink faster than length grows late in the
range (611MB left at 250, against 1.56GB at 50) and time scales
super-linearly (16.09s at 250 vs. 5.15s at 150 — roughly 3x time for <2x
length, consistent with the trunk's pairwise-attention cost), so a
meaningfully longer sequence than 250 would need checking directly rather
than assumed safe, but **mixed precision is now an optimization for
speed/headroom, not a requirement for feasibility** — the question this
section was run specifically to resolve before deciding whether to invest
in `autocast`.

## 4. fp16 produces 100% NaN coordinates — the important finding, not a setup detail

**Naive fp16 (`torch_dtype=torch.float16`, `.half()`-equivalent — the
standard recommendation for running ESMFold on modest GPUs) makes the
model return completely corrupted output, silently, at every length
tested:**

| length | positions NaN / total | plddt |
|---|---|---|
| 50 | 2,100 / 2,100 | NaN |
| 100 | 4,200 / 4,200 | NaN |
| 150 | 6,300 / 6,300 | NaN |
| 250 | 10,500 / 10,500 | NaN |

Every single coordinate, at every length. Not a precision shift, not
partial corruption — total.

**How this surfaced, because the failure mode matters as much as the
result.** The first symptom was not a NaN warning — it was an unrelated
crash in the auxiliary predicted-TM-score head:
`compute_tm`'s argmax-via-equality (`weighted == torch.max(weighted)`)
found zero matches and raised `IndexError: index 0 is out of bounds for
dimension 0 with size 0`. First attempt at a fix — upcasting just that
function's own inputs to fp32 — reproduced the identical crash, which
ruled out `compute_tm`'s own arithmetic as the cause: **the bad value
(NaN) was already present in `ptm_logits`, i.e. upstream in the fp16
trunk**, before reaching this function at all. Made the pTM/PAE heads
fail *soft* instead of crashing (return NaN, don't raise) specifically so
the forward pass could complete and `output.positions` — the thing the
fitness oracle actually needs, not the confidence score — could be
inspected directly. That inspection is what found the 100% figures above.

**The practical implication is the headline, not the debugging story**:
anyone following the standard advice ("use `.half()` on modest GPUs") and
checking only that the model *ran* — no exception, a returned object with
the right shape — would get a fitness oracle silently returning garbage
coordinates on every call. The exception from the pTM head, if hit before
any fix, is actually a mercy: it fails loud. A configuration or
transformers version where the pTM head doesn't happen to crash (plausible
— it depends on the specific NaN pattern hitting that exact equality
check) would fail **completely silently**, producing a fitness landscape
of pure noise with no error anywhere in the loop.

**Root cause not yet isolated further than "somewhere in the fp16 trunk,
before the pair representation reaches the pTM head"** — plausibly
softmax overflow or an unscaled reduction in the pairwise-attention track,
the same class of numerical issue OpenFold-family architectures are known
to need selective fp32 for (LayerNorm, softmax, certain reductions) even
under bf16/fp16 elsewhere. Not investigated further here because §3
already answered the question this was meant to resolve: fp32 covers the
full requested range, so fp16 is not required to get a working benchmark,
and is not safe to use without further work regardless.

## 5. Bottom line

- **Practical sequence-length ceiling on this hardware, fp32**: not yet
  found within the requested range — 250 residues runs with 611MB of
  headroom to spare. The real ceiling is somewhere above 250, not
  measured here.
- **fp16 is not a usable fallback as configured** — it doesn't trade
  accuracy for speed, it produces unusable output, and did so at every
  length tried. Using it for the actual search would silently poison
  every fitness evaluation.
- **Mixed precision (`torch.autocast`) is the next thing to try if
  throughput at longer sequences or larger batches becomes the
  bottleneck** — not to make ESMFold fit on this card, which fp32 already
  does across the tested range, but to recover speed once (if) a real
  search workload needs it. That's now an optimization decision with fp32
  numbers to compare against, not a blind requirement.
