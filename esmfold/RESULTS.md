# ESMFold on ece000: Setup, Precision, and Throughput

*T4 (15360 MiB, compute capability 7.5), no root, no scheduler, shared node.
`torch 2.11.0+cu128` / `transformers 5.17.0` pre-existing in
`/scratch/pcanaste/venv` (predates this work by 3 days — not newly
installed). Scripts and logs: `esmfold/` in this repo (`hpga-phase3`) —
kept here rather than a separate repo, since ESMFold is the evaluator for
the circles/blackboard architecture (`PHASE3_RESULTS.md` §8's "change the
problem" direction), not a parallel project; one history instead of two
that can drift. The revisit-rate measurement in §5 lives in
`experiments/measure_genome_revisit_rate.py`, reusing this project's
existing deterministic GA harness directly.*

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

## 3. fp32: the wall is between 250 and 300 residues

| length | elapsed | peak (torch) | peak (nvidia-smi) | headroom vs. 15360MB |
|---|---|---|---|---|
| 50 | 2.2-2.4s | 13,638 MB | 13,801 MB | 1,559 MB |
| 100 | 2.91s | 13,684 MB | 13,889 MB | 1,471 MB |
| 150 | 5.15s | 13,839 MB | 14,049 MB | 1,311 MB |
| 250 | 16.09s | 14,423 MB | 14,749 MB | 611 MB |
| 300 | — | — | — | **OOM** (tried to allocate 412MB with 305MB free) |

Pushed past the originally-requested range specifically to find the wall
rather than infer it from the trend — 611MB of headroom at 250 on a
shared, unscheduled node is thin enough that another user's allocation
could OOM a run anyway, so "probably fine" wasn't good enough. It isn't:
**300 residues OOMs immediately**, `fp32_ceiling_extended.py`. The
naive weight-size arithmetic (3.525B params x 4 bytes = 14.1GB) that
originally suggested fp32 might not fit was measured wrong at 50 residues
(actual: 13.8GB) but the right order of magnitude for where the wall
actually is — **the practical fp32 ceiling on this hardware is between
250 and 300 residues**, not "the full range with headroom to spare" as
§3 read before this was checked. Time also scales super-linearly
(16.09s at 250 vs. 5.15s at 150 — roughly 3x time for <2x length,
consistent with the trunk's pairwise-attention cost), so both the memory
wall and the time cost compound against long sequences at once.

### 3.1 Scaling is a design input, not a table row

At `pop_size=8`, `n_generations=15` (this project's standard config, e.g.
`PHASE3_RESULTS.md` §6-7) with no caching, a run needs roughly 120 fitness
evaluations (`experiments/measure_genome_revisit_rate.py` §5 below
confirms this exactly: 120/120/120 across 3 seeds). At the measured
per-call times:

| target length | 120 x elapsed | wall time |
|---|---|---|
| 50 | 120 x 2.2s | **~4.4 min** |
| 150 | 120 x 5.15s | ~10.3 min |
| 250 | 120 x 16.09s | **~32.2 min** |

**A ~7x difference in per-run wall time between the shortest and longest
target length tested, before any caching.** This is not incidental detail
to note alongside the numbers — it directly constrains which target
sequence a search can practically use: a 250-residue target costs roughly
half an hour of pure ESMFold evaluation per run, against five minutes at
50, for the same GA budget. Target-length choice for the eventual search
should be made with this table in hand, not decided first and measured
after.

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

## 5. Caching: a real but modest win, measured before building it

ESMFold is deterministic — identical sequences fold to identical
structures. An evolutionary search revisits candidates: elitism carries
the same best genome forward, and this harness's `Island._dispatch_and_collect`
re-evaluates the *full* population every generation, including unchanged
elite copies (confirmed by reading `island.py`, not assumed) — plus, as a
population converges (already well-documented for this harness, e.g.
`PHASE3_RESULTS.md` §2/§6/§7's diversity collapse), crossover and mutation
increasingly regenerate genomes already seen.

`experiments/measure_genome_revisit_rate.py` measures this directly rather
than guessing: monkeypatches `Island._dispatch_and_collect` to log every
genome evaluated in call order, at this project's standard config
(`pop_size=8`, `n_generations=15`, `genome_length=24`, `elitism=1`,
deterministic operators — free, no GPU), then replays that log against a
running "already seen" set the way a real hash cache would. Genome
content (HP-lattice symbols here vs. amino acids for the real search)
doesn't matter for this measurement — a revisit is a property of the GA's
selection/elitism/convergence dynamics, not of what the symbols mean, so
this harness is a direct, honest proxy for what an ESMFold-backed search
would revisit under identical population mechanics.

| seed | total evals | unique genomes | cache hits | hit rate |
|---|---|---|---|---|
| 0 | 120 | 83 | 37 | 30.8% |
| 1 | 120 | 82 | 38 | 31.7% |
| 2 | 120 | 77 | 43 | 35.8% |
| **mean** | **120** | — | — | **32.8%** |

(120 evaluations/run matches §3.1's arithmetic exactly — same config,
independently confirmed, not assumed.) Hit rate is not flat across a run:
it starts at 0% (generation 0, an all-random initial population, nothing
to have seen yet) and climbs as the population converges — seed 0's
generation 13-14 alone hit 4/8 and 5/8, against 0-2/8 in generations 0-3.

**A hash cache in front of ESMFold would turn roughly a third of
evaluations into free lookups, growing over the course of a run — real,
worth building, not dramatic.** Applied to §3.1's 250-residue estimate:
120 evaluations x 16.09s = 32.2 minutes uncached; at a 32.8% hit rate,
only ~81 evaluations actually need to hit the GPU, cutting that to
roughly **21.6 minutes** — about 10.6 minutes saved per run, not the
"half an hour becomes something much smaller" a naive reading of "a third
of calls are free" might suggest. The saving is real and roughly
proportional to the hit rate throughout, not concentrated late — worth
building alongside, not instead of, choosing a shorter target length in
the first place (§3.1).

## 6. Bottom line

- **fp32 practical ceiling: between 250 and 300 residues.** 250 runs with
  611MB of headroom (thin, on a shared unscheduled node); 300 OOMs
  immediately. Time cost is the other constraint, independent of memory —
  §3.1's ~7x wall-time spread (5→32 min for the same GA budget, 50 vs.
  250 residues) should drive target-length choice before the memory wall
  does, since it binds well before 300 residues would.
- **fp16 is not a usable fallback as configured** — it doesn't trade
  accuracy for speed, it produces unusable output (100% NaN coordinates),
  and did so at every length tried. Using it for the actual search would
  silently poison every fitness evaluation.
- **Mixed precision (`torch.autocast`) is worth trying next if throughput
  at longer sequences or larger batches becomes the bottleneck** — not to
  make ESMFold fit on this card (fp32 already does, up to the 250-300
  wall), but to push that wall out and recover speed, if a real search
  workload needs sequences longer than fp32 can hold. That's now an
  optimization decision with fp32 numbers and a caching estimate to
  compare against, not a blind requirement.
- **A hash cache is worth building**: ~33% of evaluations would be free
  lookups at this project's standard GA config, growing as the population
  converges. Real savings, not a substitute for keeping the target length
  short (§3.1) — the two compound rather than trade off against each
  other.
