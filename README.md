# Phase 2 — LLM-Based Genetic Operators

Self-contained sibling of `Phase 1/`, structured the same way (`hpga/`,
`experiments/`, `analysis/`, `results/`). This folder exists so Phase 1 stays
untouched and reproducible while Phase 2 substitutes LLM calls for the
deterministic crossover/mutation operators.

**Not a git repository** — this project has no `.git` at the time of writing,
so there is no commit to point at. Identity between the two folders is
recorded below as SHA-256 file hashes instead, taken at the moment `Phase 2/`
was split out of `Phase 1/` (2026-08-30).

## What changed vs. Phase 1

Originally only `hpga/operators.py` (see below). As of the 3D migration
(advisor requirement, see "3D migration" section below), `hpga/hp_model.py`
and `hpga/config.py` also intentionally differ — Phase 2 now models the
simple cubic lattice (3D), Phase 1 stays the untouched 2D square-lattice
baseline. `island.py`, `worker.py`, and `instrumentation.py` are still
byte-identical to Phase 1 and dimension-agnostic (they call
`evaluate_fitness`/`random_population` generically, with no coordinate
assumptions), so the DIS/GA timing-harness comparison those files support is
unaffected by the dimensionality change. `experiments/` and `analysis/` were
copied too, for a complete standalone harness, and are otherwise identical
aside from the new 3D-specific scripts noted below.

`hpga/operators.py` keeps both implementations. Which one runs is chosen at
call time by the `HPGA_OPERATOR_MODE` env var (`"deterministic"`, the
default — identical output to Phase 1 — or `"llm"`), read fresh inside
`next_generation()` on every call, not cached at import. This means:

- island.py's call site (`next_generation(...)`) needed no signature change
  and has none.
- A single process can run a deterministic run immediately followed by an
  LLM run (or vice versa) for a paired comparison, which is how
  `experiments/run_phase2_pilot.py` works.

In `"llm"` mode, crossover and mutation are calls to a local Ollama server
(`gemma4:12b` by default), one call per `crossover()`/`mutate()` invocation —
the same call cardinality as the deterministic path, so LLM latency
substitutes for near-zero deterministic compute rather than also multiplying
call count. Selection (tournament) and elitism are always deterministic; only
"how do these genes recombine/mutate" is delegated to the model. Malformed
model output is retried (`HPGA_LLM_MAX_RETRIES`, default 2 extra attempts)
and, on repeated failure, falls back to the deterministic operator for that
call rather than crashing the run — this path is exercised deliberately in
`experiments/`, see below.

Every LLM prompt/response is logged as one JSON line per call to
`results/raw/llm_operator_calls_<run_id>.jsonl` (`HPGA_RUN_ID` env var, or a
timestamp+PID default), so operator behaviour can be audited after the fact.
Two such logs, produced during the initial pilot before this split, were
moved here from `Phase 1/results/raw/`:
`llm_operator_calls_1788052574_11252.jsonl` (an isolated one-crossover /
one-mutate sanity check) and `llm_operator_calls_1788052620_7788.jsonl` (the
first paired pilot run, 27 calls, 27/27 valid on first attempt).

## Files specific to Phase 2

- `hpga/operators.py` — the only modified file (see above).
- `experiments/probe_ollama_concurrency.py` — measures whether concurrent
  requests to the local Ollama daemon actually parallelise (they don't,
  linearly: throughput goes 1.00x/1.49x/1.96x/2.28x relative to K=1 as K goes
  1/2/4/8 — sub-linear, not flat, a deployment artifact of this Ollama
  instance, not of the architecture).
- `experiments/run_phase2_pilot.py` — paired short run, deterministic vs.
  LLM operators, same config otherwise.
- `experiments/run_phase2_n_sweep.py` — small-scale (`pop_size=8`) worker-count
  sweep under both operator modes, testing whether N still matters once the
  GA phase runs LLM calls.
- `experiments/probe_latency_vs_tokens.py` — fits `latency_s ~= a + b*tokens_in
  + c*tokens_out` by least squares against `results/raw/latency_vs_tokens.jsonl`.
- `experiments/run_diff_style_probe.py` — paired comparison of the two
  `HPGA_LLM_PROMPT_STYLE` values (see below): does asking the model for only
  the changed genome positions, instead of the full genome, cut output tokens
  and latency by roughly the amount the fitted model predicts.
- See `results/PHASE2_RESULTS.md` for what these produced.

## Byte-identity record (`hpga/`), SHA-256, as of the initial LLM-operator split (2026-08-30)

| file | Phase 1 | Phase 2 |
|---|---|---|
| `__init__.py` | `4847a52b53b9363932d85feaa0e4aa5d40d23229897b2ec5af891426595719d2` | identical |
| `config.py` | `e1b7bc3a564ee6e844275e71938199e67977fac84cfeac43809206bcaf3f1dfa` | identical at split time; **intentionally differs as of the 3D migration**: `82150569b3da0c4bed66733e9735f747a7c8f5107d376c069d85983d923b358d` (adds `BENCHMARK_SEQUENCE_3D`/`BENCHMARK_OPTIMAL_ENERGY_3D`, existing 2D constants untouched) |
| `hp_model.py` | `37674dfdacc8da27c28775ab5e463d904e1290c027ac4e70c85e4ef7e7c03340` | identical at split time; **intentionally differs as of the 3D migration**: `94c16274dee157d340c4635324c12bc7ddd7ae152d9e2698c07454b170c917f1` (2D square lattice -> 3D simple cubic lattice; see "3D migration" below) |
| `instrumentation.py` | `f7e46b2f7ffff95682d01ca9399dd1c3dbef96df60043143bc812f263d0fbad6` | identical |
| `island.py` | `bfaa49f4cb9715ffdd6899cda71d6d76d3db181ffda1c6b461cf49cc90036731` | identical |
| `worker.py` | `13e0fe082765b52f97de25ad6a976cb07983984421e2f75afe61bc9c5132fe47` | identical |
| `operators.py` | `2d9930388127845892b8a5c5830e67f1d97ed4c053385a4846b6bf8bd57febf4` | `7a7e0ae42d2471a1dc7fad7312a4c1d63b31c6b25bc78b566d1a9884b7e7928a` (intentionally differs since the initial split) |

`experiments/{run_benchmark_validation,run_smoke,run_sweep,run_sweep_repeats}.py`
and `analysis/{model,plots,master_overhead}.py` were also verified identical
to their Phase 1 counterparts at split time and are unaffected by the 3D
migration (they don't touch `hp_model.py` directly).

Re-verify the *timing-harness* files (still required to match) with:

```bash
for f in island.py worker.py instrumentation.py __init__.py; do
  diff -q "../Phase 1/hpga/$f" "hpga/$f"
done
```

`config.py` and `hp_model.py` are now expected to differ from Phase 1 (3D
migration, flagged below) and are excluded from that loop on purpose.

**If any of `island.py`, `worker.py`, or `instrumentation.py` ever needs to
change in Phase 2, that breaks the comparison's validity and must be flagged
before the change is made, not after.**

## 3D migration (advisor requirement)

`hpga/hp_model.py` was migrated from the 2D square lattice to the 3D simple
cubic lattice. Phase 1 is untouched and remains the 2D baseline; Phase 2 is
now 3D-only (there is no dimensionality switch — this is a full replacement
of the fitness/geometry module, not a mode flag).

**Model.** 6 neighbours per lattice cell. The genome is still a
relative-move string, now 5 symbols instead of 3: `STRAIGHT, LEFT, RIGHT,
UP, DOWN` (codes 0–4, replacing 2D's `STRAIGHT, LEFT, RIGHT`). The 6th
direction (directly reversing the current heading) is excluded from the
encoding for the same reason 2D has 3 symbols instead of 4: reversing always
places the new residue on top of residue *i-1*, a guaranteed collision, so
it's removed by construction rather than checked at runtime. Orientation is
tracked as a `(forward, up)` frame of axis-aligned unit vectors (right =
`cross(forward, up)`, derived, not stored); each move is an exact 90-degree
rotation of the frame, implemented as integer cross products (`v_rot = axis
x v`, the closed form of Rodrigues' formula at 90 degrees) — no floating
point, so the frame never drifts off-lattice. Fitness is unchanged in kind:
count of H-H contacts adjacent on the lattice but non-consecutive in the
chain, with the same graded collision penalty (`-collision_penalty -
n_collisions`) that gives the GA a gradient out of invalid folds instead of
a flat penalty plateau. `island.py`/`worker.py`/`instrumentation.py` needed
no changes — they call `evaluate_fitness`/`random_population` generically
and never touch coordinates directly.

**Validation.** Same method as Phase 1's 2D validation
(`experiments/run_benchmark_validation.py`): run the deterministic GA
(`HPGA_OPERATOR_MODE` unset — no LLM operators involved) against a benchmark
sequence with a known optimal energy, across several seeds, and check
whether it reaches or approaches that optimum.
`experiments/run_benchmark_validation_3d.py` does this for the 3D model.

Benchmark: **Mann, M., Will, S. & Backofen, R. (2008). "CPSP-tools — exact
and complete algorithms for high-throughput 3D lattice protein studies."
BMC Bioinformatics 9, 230.** https://doi.org/10.1186/1471-2105-9-230 —
Table 2, sequence S1: `HHHHHPHHPHPHPHPHPHPHHHHHHPH` (n=27). Plain
(backbone-only) HP model, unrestricted simple cubic lattice — the same
model this file implements. **E\*=-22 (22 non-consecutive H-H contacts) is
proven optimal, not heuristic best-known**: CPSP-tools' branch-and-bound
(HPstruct) exhaustively proves no structure of this sequence exceeds 22
HH-contacts, and reports degeneracy=1 (a unique optimal structure up to
lattice symmetry) — a stronger and more citable guarantee than the
originally-considered alternative (the Yue et al. 1995, PNAS 92:325-329,
48-residue "Harvard sequences," whose *designed* target energies were not
independently re-extractable from available sources during this search —
see the gap note below).

**Gap vs. the base paper's actual 3D model, stated plainly (per request,
not papered over):** Xue et al. (NoCS 2014, this project's base paper) use
the **HP side-chain model (HPSC, Benitez & Lopes 2009)** for their
Unger273d/Dill benchmarks — each residue occupies two lattice points (a
backbone bead plus a side-chain bead), with its own contact/energy scheme.
What's implemented and validated here is **plain 3D HP** (backbone-only,
this file) — a real step closer to HPSC than the 2D model was (same lattice
dimensionality, same 6-neighbour cubic geometry), but not the same model.
A search during this migration did not turn up a plain-3D-HP-with-published-optimum
benchmark that is *also* directly comparable to Xue et al.'s numbers,
because Xue et al.'s own benchmark values are HPSC-specific and not
meaningful for a backbone-only model. If Phase 3+ needs a validated HPSC
model to compare against Xue et al.'s actual benchmark numbers, that is a
separate, larger piece of work (implementing the side-chain lattice
placement and HPSC's contact/energy rules) than this migration, and should
be scoped separately rather than assumed to fall out of the plain-3D model
validated here.

**Results** (`experiments/run_benchmark_validation_3d.py`, `pop_size=256`,
`n_generations=500`, deterministic operators, 5 seeds):

| seed | final contacts | g0 | g25 | g50 | g100 | g200 | g300 | g400 | g499 |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 19 | 11 | 15 | 16 | 17 | 18 | 18 | 18 | 19 |
| 1 | 18 | 6 | 14 | 17 | 17 | 18 | 18 | 18 | 18 |
| 2 | 19 | 9 | 16 | 17 | 19 | 19 | 19 | 19 | 19 |
| 3 | 20 | 6 | 14 | 18 | 20 | 20 | 20 | 20 | 20 |
| 4 | 20 | 7 | 15 | 15 | 18 | 20 | 20 | 20 | 20 |

Best across 5 seeds = 20/22 contacts, gap = 2. **The GA did not reach the
proven optimum in any seed. The validation did not pass.** This falls short
of Phase 1's 2D validation (Unger & Moult n=20, E\*=-9: 3/5 seeds reached
the exact optimum, the other 2 landed 1 contact short, converged by
generation ~25–50 — `Phase 1/results/PHASE1_RESULTS.md` §2) on the same
operator/population budget.

A plausible explanation (larger 5-symbol alphabet, a degeneracy=1 target
being intrinsically harder to hit exactly than the 2D benchmark's easier
landscape) is not a demonstration. On its own, this result cannot
distinguish "model correct, GA short of budget" from "a subtle geometry bug
the migration's own unit tests didn't catch." That distinction rests on the
correctness checks below, not on the benchmark run itself.

**What was checked, and what each check does and does not establish:**

1. **Rotation composition identities** (run during the migration, before
   this validation): 4×LEFT and 4×UP each return the frame to identity,
   LEFT/RIGHT and UP/DOWN each cancel, and the frame stays exactly
   orthonormal and axis-aligned over 1000 random moves. Establishes the
   turn logic is internally self-consistent and never drifts off-lattice.
   Does **not** establish that any single turn rotates in the physically
   intended direction — a systematically flipped sign convention would
   pass all of these identities too (though see point 4: for HP contact
   counting specifically, a global sign/reflection convention is provably
   harmless — see the code comment in `hp_model.py`).
2. **Yaw-plane 2D-reduction**: a genome using only STRAIGHT/LEFT/RIGHT
   traces the exact same shape (up to labelled axes) as the already-validated
   2D model. Confirms the LEFT/RIGHT half of the encoding is consistent
   with a working reference implementation.
3. **Pitch-plane 2D-reduction** (added during this check, symmetric to #2):
   a genome using only STRAIGHT/UP/DOWN is confirmed to stay confined to a
   single plane and trace the identical square shape as the LEFT×4 case.
   Closes the gap #2 left open — UP/DOWN was previously verified only by
   the algebraic identities in #1, never against any shape-based reference.
4. **Independent contact-counter cross-check**: a from-scratch pairwise-Manhattan-distance
   contact counter (no neighbor-offset table, no dict, not sharing any code
   with `fitness_from_coords`) was run against `fitness_from_coords` on
   2000 random genomes. **0/2000 mismatches.** Strong evidence the
   contact/collision-counting logic itself is correct, independent of
   whatever coordinates `genome_to_coords` hands it.
5. **The check originally asked for — an external, independently-proven
   ground truth — was attempted and is currently blocked, not skipped.**
   CPSP-tools' own optimal structure for the S1 benchmark was going to be
   fetched from their live HPstruct solver (they publish structures, not
   just energies) and scored through `fitness_from_coords` directly. The
   job was submitted twice (jobID `3058336`, waited 15 minutes; a retry,
   jobID `3478281`, spot-checked ~80s later) and **neither job ever
   advanced past "Submitted & Queued"** — a stalled server-side job
   runner on CPSP-tools' end, not a fluke of one submission. The letter
   convention needed to decode their "absolute move string" output (taken
   verbatim from their own `absoluteMoveToPDB.js`, not guessed) and the
   comparison script (`experiments/verify_3d_benchmark_structure.py`) are
   ready to run the moment either job completes, or a fresh one does — this
   is worth re-attempting, not a dead end.
6. **The check actually completed in place of #5** — fully self-contained,
   no external dependency (`experiments/verify_3d_small_instance_exact.py`):
   on a small (n=8) instance, a from-scratch brute-force solver (raw
   6-direction DFS over self-avoiding walks, independent contact counter,
   zero shared code with `hpga/`) computed the true optimal HH-contact
   count. Exhaustively enumerating hp_model's *entire* reduced-move genome
   space for the same sequence (5^6 = 15625 genomes, via the actual
   production `evaluate_fitness`) reached exactly the same value. **Match.**
   This proves, for this instance, that the full genome → coordinates →
   fitness pipeline reaches and correctly scores the true global optimum —
   not merely internally consistent, but correct against independently-computed
   ground truth. A bug in the per-step rotation or contact logic would be
   expected to surface here too, since it's the same code applied to fewer
   residues, not different code.

**Bottom line, stated plainly:** points 1–4 and 6 are real, converging
evidence against a geometry/contact-logic bug, including one instance
(point 6) checked against genuine independent ground truth rather than only
internal consistency. But point 5 — the specific external check this
project's benchmark citation is based on — did not complete, so the 3D
model's correctness on the actual 27-residue S1 benchmark is not proven,
only well-supported by the smaller-scale and structural checks above. The
GA's 20/22 shortfall on `run_benchmark_validation_3d.py` should be read as
*most likely* a search-budget/landscape-difficulty gap rather than a model
bug, given the above — not as a settled fact. Re-running
`verify_3d_benchmark_structure.py` against a completed CPSP-tools job (retry
jobID `3478281`, or a fresh submission) remains the outstanding step that
would fully settle it.

## Environment

Own venv (`.venv/`), not shared with Phase 1, pinned via `requirements.txt`
(frozen from Phase 1's venv at split time, plus `ollama`/`matplotlib`/`numpy`
already present there). Requires a local Ollama server
(`http://localhost:11434` by default) with `gemma4:12b` pulled.

Relevant env vars (all read by `hpga/operators.py`, all optional):

| var | default | purpose |
|---|---|---|
| `HPGA_OPERATOR_MODE` | `deterministic` | `deterministic` or `llm` |
| `HPGA_LLM_PROMPT_STYLE` | `full` | `full` (model restates the entire child/mutated genome) or `diff` (model emits only the positions that change from a stated reference genome; applied on top of a copy of it). Only meaningful when `HPGA_OPERATOR_MODE=llm`. |
| `HPGA_LLM_MODEL` | `gemma4:12b` | Ollama model tag |
| `HPGA_OLLAMA_HOST` | `http://localhost:11434` | Ollama server URL |
| `HPGA_LLM_TEMPERATURE` | `0.7` | sampling temperature |
| `HPGA_LLM_NUM_CTX` | `4096` | context window |
| `HPGA_LLM_KEEP_ALIVE` | `30m` | Ollama model keep-alive |
| `HPGA_LLM_TIMEOUT_S` | `120` | per-request HTTP timeout |
| `HPGA_LLM_MAX_RETRIES` | `2` | retries beyond the first attempt before falling back |
| `HPGA_RUN_ID` | `<timestamp>_<pid>` | log filename suffix |
| `HPGA_LLM_LOG_PATH` | (derived from `HPGA_RUN_ID`) | explicit log file override |

Agents/communication (`hpga/agents.py`, see `PHASE2_RESULTS.md` §8 -- new
architecture on top of the above, off by default):

| var | default | purpose |
|---|---|---|
| `HPGA_AGENTS_ENABLED` | `0` (off) | `1` adds `HPGA_N_AGENTS` agent-produced genomes on top of the ordinary crossover/mutate fill each generation. Only meaningful when `HPGA_OPERATOR_MODE=llm`. |
| `HPGA_N_AGENTS` | `2` | number of agents per island (`A`) |
| `HPGA_COMM_INTERVAL` | `5` | generations between central-node communication rounds (fires when `generation > 0 and generation % interval == 0`); set past `n_generations` to keep agents active with communication effectively disabled |
| `HPGA_AGENT_ROLES` | alternating `explore`/`refine` | comma-separated role list of length `HPGA_N_AGENTS`, e.g. `explore,refine,explore`; falls back to the default if the length doesn't match |
| `HPGA_ISLAND_ID` | `0` | island id tag on agent/diversity log records (no multi-island driver exists in Phase 2 yet) |
| `HPGA_LOG_DIVERSITY` | `0` (off) | `1` logs per-generation mean pairwise Hamming distance to `results/raw/diversity_<run_id>.jsonl`, independent of `HPGA_AGENTS_ENABLED` |
| `HPGA_AGENT_LOG_PATH` | (derived from `HPGA_RUN_ID`) | explicit override for `results/raw/agent_messages_<run_id>.jsonl` |
| `HPGA_DIVERSITY_LOG_PATH` | (derived from `HPGA_RUN_ID`) | explicit override for the diversity log |
