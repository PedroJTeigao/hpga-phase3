# Move class: does a different kind of move pay more per evaluation?

**Direction-finding, not a result.** Three seeds per arm. Under the fixed rule used throughout this project (X beats Y only if X is higher in *every* seed), the smallest two-sided sign-test p reachable with 3 seeds is 2/2³ = **0.25**. Nothing in this document is significant on its own. Every "beats" below is the fixed rule applied to three seeds, and should be read as a pointer toward what to test with more seeds, not as a finding.

## The question, and why it is asked this way

Evidence 4 of `results/OPERATOR_DOES_NOT_MATTER.md` measured the **greedy oracle's** candidate set (arm O). At each breeding step, each oracle slot tried 8 single-position substitutions of its own tail-slot genome, scored every one with the real fitness function, and kept the best. In 53–61% of steps none of the 8 beat the genome it already held. When one did, the mean gain was 0.0075–0.0206 TM. That candidate set was single-position **by construction of the oracle**. Evidence 4 was not measured on the deterministic GA baseline (arm B), whose mutation operator is a different move class (below).

Evidence 4 is a statement about the move class, not about who chooses the move. So this experiment holds the chooser fixed (no LLM anywhere; `HPGA_OPERATOR_MODE=deterministic`) and varies only the move class. The measured quantity is payoff per move: how often one move improves on the genome it was applied to, and by how much when it does.

## Design

Everything is arm B of `experiments/run_sequence_ga_comparison.py`, unchanged:

- population 16, 20 evaluated populations (generation 0 = random start, so 19 breeding steps), tournament 3, elitism 2
- crossover 0.9 (single proportional cut), length bounds [30, 80]
- `Random(seed)` drives everything, and every arm starts from the same generation-0 population per seed
- fitness = TM-score of the ESMFold fold against 7UR7 chain A, through the same per-run cache
- seeds 0, 1, 2

The driver (`experiments/run_move_class.py`) mirrors `run_ga`'s deterministic loop line for line. It replaces only the active model's `deterministic_mutate`, for the duration of one run, with the arm's move (`hpga/move_class.py`). `island.py`, `worker.py` and `instrumentation.py` are untouched, and nothing in `hpga/` other than the new `move_class.py` changed.

| arm | move applied to each of the 14 non-elite children per breeding step |
|---|---|
| **S1** | exactly one substitution: one position uniform, new letter uniform over the other 19 |
| **S2** | one k-position substitution: k uniform in 2–5, positions drawn without replacement, each new letter uniform over the other 19 |
| **S3** | one segment replacement: span length uniform in 5–15, start uniform over the starts where the whole span fits, span refilled with letters uniform over all 20 |
| **S4** | one indel: delete or insert with equal probability, span 1–5. A deletion's start is uniform over the starts where it fits; an insertion goes at a gap uniform over 0..len with fresh letters. A draw that would leave [30, 80] is redrawn (9 of 798 moves). |
| **B** (reference) | the existing deterministic operator: **per-site substitution with p = 0.05**, new letter uniform over all 20 (so it can equal the old one). That is about 3 substitutions per child on ~60-residue genomes, with anywhere from 0 to 6+, and 4.1% of B's moves changed nothing. |

**S1 is a new arm, not the baseline.** The deterministic baseline was never single-position. S1 was added so that S1–S4 each apply exactly one move of their class per child, which makes them comparable as move classes. Single-position gets its own arm because Evidence 4's oracle candidates were single-position. B is reported as a fifth reference row. It overlaps S2 in how many sites it touches, but it is not one move per child.

## Verification: B reproduces the published baseline, and measuring changed nothing

- **B reproduces arm B exactly, in all three seeds.** `results/raw/move_class_B_seed{0,1,2}.json` match `results/raw/sequence_ga_cmp_B_seed{0,1,2}.json` on distinct-fold count, the best-so-far curve and its generation tags, and all 20 generations' populations and fitnesses, compared as exact JSON values. Seed 0 was run and checked alone before any other arm was launched. This shows the wrapper plumbing is exact, and that S1–S4 differ from B only in the move.
- **Measurement is inert, asserted twice per run.** The base fitnesses a payoff needs (the post-crossover child, which the GA never folds unless it is an unchanged parent) were folded only after the GA had finished. They went into a separate cache and a separate file, `results/raw/move_class_bases_<arm>_seed<n>.json`, never into the run's own file. The two checks:
  1. A SHA-256 digest of the GA's outputs (distinct count, best-so-far curve and tags, every population and fitness list) was taken before any measurement fold and is identical after.
  2. The GA was replayed from the seed with move recording off, with fitness served only from the run's own cache; a genome the run never folded would have raised. The replay's digest is identical to the run's.

  Both checks passed for all 15 runs (`summary.inertness` in each run file). For B, the published-run match is a third, external check of the same thing.
- **What measuring cost.** Measurement folds are extra GPU time, charged to nothing in the search:

| arm | extra folds per seed (0 / 1 / 2) | measurement share of total fold time per seed |
|---|---|---|
| S1 | 95 / 158 / 144 | 26.6% / 35.4% / 33.6% |
| S2 | 210 / 174 / 179 | 41.9% / 39.8% / 38.9% |
| S3 | 188 / 146 / 138 | 39.8% / 32.4% / 32.6% |
| S4 | 185 / 192 / 183 | 40.0% / 41.2% / 39.5% |
| B | 168 / 172 / 161 | 43.2% / 38.4% / 36.6% |

Across all 15 runs that is 2,493 measurement folds and 1.74 GPU-hours, against 2.90 GPU-hours of the GA's own folds: 37.5% of all fold time. S1 needed the fewest because one substitution applied to an unchanged parent (a no-crossover child) has a base already in the run's cache.

## 1. Best fitness at the common evaluation count

Each seed is read at `n_cut`, the smallest distinct-fold count any of the five arms reached in that seed: 278, 276 and 280 for seeds 0, 1 and 2. B sets `n_cut` in every seed, because its unchanged children are cache hits. S1–S4 reached 281–282 distinct folds.

| arm | seed 0 | seed 1 | seed 2 | mean |
|---|---|---|---|---|
| S1 single substitution | 0.4031 | 0.3875 | 0.4816 | 0.4241 |
| S2 2–5 substitutions | 0.4107 | **0.6522** | 0.4175 | **0.4935** |
| S3 segment 5–15 | 0.3910 | 0.4193 | **0.5265** | 0.4456 |
| S4 indel 1–5 | **0.4379** | 0.4770 | 0.4750 | 0.4633 |
| B per-site p=0.05 (reference) | 0.4203 | 0.5025 | 0.5040 | 0.4756 |

**Only one pair is ordered in every seed: B beats S1** (0.4203 > 0.4031, 0.5025 > 0.3875, 0.5040 > 0.4816). No move class beats any other move class in every seed. The best arm changes with the seed: S4 in seed 0, S2 in seed 1, S3 in seed 2.

S2's lead in the mean comes from one seed. Seed 1's 0.6522 is above every final best cited in `OPERATOR_DOES_NOT_MATTER.md` (0.38–0.60). It is a sustained climb in four steps over generations 8–18, not a single lucky fold: the population best goes 0.39 → 0.52 (generation 8) → 0.56 (11) → 0.62 (14) → 0.65 (18). The best genome is 55 edits from the native sequence. The largest single step was one 4-position S2 move at step 13 that took a 0.3515 base to 0.6162. In seeds 0 and 2, S2 is below B.

## 2. Fitness trajectory, mean over seeds, against evaluations spent

![Best-so-far vs distinct evaluations](move_class.png)

| evaluations | S1 | S2 | S3 | S4 | B |
|---|---|---|---|---|---|
| 16 | 0.3106 | 0.3106 | 0.3106 | 0.3106 | 0.3106 |
| 48 | 0.3444 | 0.3293 | 0.3867 | 0.3540 | 0.3737 |
| 96 | 0.3755 | 0.3897 | 0.3942 | 0.3936 | 0.3983 |
| 144 | 0.3900 | 0.4343 | 0.4259 | 0.4286 | 0.4284 |
| 192 | 0.3968 | 0.4565 | 0.4286 | 0.4402 | 0.4535 |
| 240 | 0.4118 | 0.4818 | 0.4394 | 0.4528 | 0.4695 |
| 276 | 0.4241 | 0.4935 | 0.4456 | 0.4633 | 0.4756 |

(Every 16 evaluations: `results/MOVE_CLASS_TABLES.md` §2. Every evaluation: `results/move_class_summary.json`, `trajectory_full`.) The first 16 evaluations are the shared generation 0. S1 is the lowest mean curve from about 100 evaluations on. The other four stay within about 0.03 of each other until S2's seed-1 climb separates it after about 200 evaluations.

## 3. Payoff per move: share that improve on their base, and mean gain when they do

A move is one mutation call on one post-crossover child. It improves if the child's fitness is strictly above its base's. The table covers all 19 breeding steps, with 266 moves per run.

| arm | improved: seed 0 / 1 / 2 | **improved, pooled** | gain if improved: seed 0 / 1 / 2 | **gain if improved, pooled** | expected positive gain per move (share × gain), pooled |
|---|---|---|---|---|---|
| S1 | 24.4% / 33.5% / 34.2% | **30.7%** | 0.0230 / 0.0204 / 0.0227 | **0.0220** | 0.0067 |
| S2 | 34.2% / 33.8% / 34.6% | **34.2%** | 0.0293 / 0.0487 / 0.0252 | **0.0343** | 0.0118 |
| S3 | 33.1% / 31.6% / 20.3% | **28.3%** | 0.0296 / 0.0332 / 0.0366 | **0.0326** | 0.0092 |
| S4 | 34.6% / 39.5% / 40.2% | **38.1%** | 0.0253 / 0.0360 / 0.0311 | **0.0310** | 0.0118 |
| B | 28.9% / 33.5% / 31.6% | **31.3%** | 0.0199 / 0.0339 / 0.0358 | **0.0302** | 0.0095 |

Orderings that hold in every seed:

- **Share improved: S4 is higher than every other arm (S1, S2, S3 and B) in every seed.** S2 is higher than S1, S3 and B in every seed.
- **Gain if improved: S2, S3 and S4 are each higher than S1 in every seed.** No other pair is ordered in every seed.

So, per move, single-position substitution is at the bottom on both numbers. Its gain when it improves is the lowest of the four move classes in every seed. Its improvement rate is below S2 and S4 in every seed and above only S3's pooled figure. A single random substitution to a crossover child pays about 0.022 TM when it helps. A 2–5-site substitution, a segment or an indel pays about 0.031–0.034. Indels help most often.

Most moves hurt in every arm. 62–72% of moves are worse than their base, and the mean gain over all moves is negative everywhere: S4 −0.020, S1 −0.022, S2 −0.025, B −0.033, S3 −0.038. No S1–S4 move landed exactly on its base's fitness. Only B, whose per-site draw can change nothing, has equal-fitness moves (4.1%).

**How this relates to Evidence 4.** S1's gain when it improves (0.020–0.023) is of the same order as the oracle's gain when it moved (0.0075–0.0206). But these are not the same statistic, so they should not be read as a replication:

- The oracle's figure is the best of 8 candidates, applied to a tail-slot genome it had already been climbing, with greedy acceptance.
- S1's figure is one random substitution applied to a fresh crossover child, which is often less fit than either parent and so easier to improve on.

What the two share is magnitude: a single substitution buys roughly 0.02 TM when it helps, in both settings.

## 4. The same two numbers by generation thirds

Thirds are by breeding step: early = steps 0–5, middle = 6–11, late = 12–18. The children are evaluated in generations 1–6, 7–12 and 13–19. Each cell is pooled over seeds, as improved share / gain if improved. Per-seed values are in `results/move_class_summary.json`, `payoff_thirds`.

| arm | early | middle | late |
|---|---|---|---|
| S1 | 40.9% / 0.0223 | 23.0% / 0.0193 | 28.6% / 0.0234 |
| S2 | 43.3% / 0.0303 | 31.0% / 0.0339 | 29.3% / 0.0399 |
| S3 | 37.3% / 0.0351 | 28.2% / 0.0330 | 20.7% / 0.0284 |
| S4 | 44.0% / 0.0298 | 34.9% / 0.0306 | 35.7% / 0.0327 |
| B | 38.9% / 0.0275 | 25.0% / 0.0309 | 30.3% / 0.0328 |

- **Every arm improves most often early.** Random-start genomes are easy to improve on. The improvement rate falls by the middle third in every arm.
- **S4 has the highest pooled improvement rate in every third,** and it holds up best late (35.7%, against 20.7–30.3% for the others).
- **S3 declines steadily** (37% → 28% → 21%, with seed 2 late at 11.2%). Replacing 5–15 residues of a genome that selection has already shaped mostly destroys what was there. Its gain when it does improve also shrinks late.
- **S2's gain if improved grows** (0.030 → 0.034 → 0.040) while its improvement rate falls. Late in the run, a multi-site substitution that helps helps by more.
- **S1's gain if improved is the lowest of the five arms in every third,** staying near 0.02 throughout. Its improvement rate is the lowest in the middle third, second-lowest late and middling early.

## 5. S4 genome length over time

All evaluated S4 genomes per generation, pooled over the three seeds. Per-generation min, quartiles, max and per-seed means for all 20 generations are in `results/MOVE_CLASS_TABLES.md` §5.

| generation | min | q1 | median | q3 | max | mean | per-seed mean (0 / 1 / 2) |
|---|---|---|---|---|---|---|---|
| 0 | 30 | 42.0 | 54.0 | 62.0 | 79 | 53.1 | 52.0 / 52.7 / 54.8 |
| 2 | 47 | 58.0 | 65.0 | 69.8 | 79 | 63.8 | 62.6 / 64.7 / 64.1 |
| 5 | 49 | 62.0 | 66.5 | 72.0 | 79 | 66.2 | 65.0 / 64.8 / 69.0 |
| 10 | 57 | 64.0 | 67.0 | 71.0 | 80 | 67.3 | 66.9 / 62.9 / 71.9 |
| 15 | 58 | 63.0 | 68.0 | 71.8 | 80 | 67.9 | 65.1 / 64.9 / 73.6 |
| 19 | 57 | 63.0 | 67.5 | 75.0 | 80 | 68.2 | 64.5 / 64.5 / 75.6 |

- **Length rises fast, then plateaus.** It goes from the uniform [30, 80] start (mean 53) to about 64 within two generations, then drifts slowly up to about 68. The short genomes are selected out early: the minimum is ≥ 44 from generation 2 on. The native reference is 63 residues.
- **Seed 2 keeps growing,** reaching a mean of 75.6 at generation 19, close to the 80-residue upper bound.
- **Insertions helped more often than deletions:** 42.2% of 403 insertions improved, against 33.9% of 395 deletions, with the same gain when they did (0.0309 vs 0.0312).
- **Drift is not S4-specific.** Crossover's proportional cut also changes length, and every arm drifts upward. The final-generation mean length per seed was 59 / 67 / 63 for S1, 54 / 64 / 68 for S2, 57 / 59 / 67 for S3, 65 / 65 / 76 for S4 and 59 / 64 / 72 for B. S4 ends longest in seeds 0 and 2, but by a few residues, not a different regime.

## What this says, and what it does not

**Move class changes per-move payoff in a direction that is consistent across seeds.** In all three seeds:

- indels (S4) improve on their base more often than every other arm;
- 2–5-site substitutions (S2) improve more often than S1, S3 and B;
- every larger move class (S2, S3, S4) gains more than single substitution when it improves.

Single-position substitution is the weakest move class on per-move payoff, and the per-site baseline B beats S1 on final fitness in every seed. That fits Evidence 4's reading that single-position edits are a weak move here. It also shows the weakness is not unique to the oracle's setting: one random substitution to a crossover child is the least productive move tested.

**It did not turn into a fitness ordering among move classes.** At the common budget no move class beats any other in every seed. Per-move payoff is one input into search quality. The best at n_cut also depends on the rare large jumps (S2 seed 1's +0.265 move), which three seeds cannot average out. The difference between the best and worst per-move expected gain (0.0118 vs 0.0067 TM per move) is small next to the between-seed spread of final fitness (0.39–0.65).

**Caveats specific to this design:**

- **Bases are not matched across arms.** Each arm's moves were applied to its own population's crossover children, so part of any payoff difference is a difference in what was being mutated. Generation 0 and the first selected pair of step 0 are shared, and everything after diverges. A matched-base design (every move class applied to the same set of bases, off-budget) would isolate the move class exactly. That would be the natural follow-up if the per-move ordering above is worth pinning down.
- **One move per child, one mutation rate.** S1–S4 ignore the 0.05 rate by design. Whether two S4 moves per child, or a mix of S2 and S4, does better was not tested.
- **"Improves on its base" is measured against the post-crossover child,** not against either parent. A crossover child is often worse than both parents, so these improvement rates are not rates of beating the parents.
- **Same scope as every other result in this project:** one target (7UR7 chain A, 63 residues), one fitness function, population 16, 20 generations, about 280 folds per run, three seeds.

**Suggested direction, not a conclusion:** if move class is taken further, S4 (highest improvement rate in every seed and every third) and S2 (highest gain when it improves, rising late) are the candidates to test with more seeds, and S1 is the one to drop.

## Files

- `hpga/move_class.py`: the five move functions
- `experiments/run_move_class.py`: the driver, including the verification and measurement steps
- `experiments/summarize_move_class.py`: generates `results/MOVE_CLASS_TABLES.md`, `results/move_class_summary.json` and `results/move_class.png`
- `results/raw/move_class_{S1,S2,S3,S4,B}_seed{0,1,2}.json`: the 15 runs, in the same schema as `sequence_ga_cmp_*`, plus per-generation `lengths`
- `results/raw/move_class_bases_{arm}_seed{n}.json`: every move with its base and child genomes and fitnesses (measurement only)
- `results/raw/move_class_run.log`, `results/raw/move_class_status.json`: the 14-run launch
- `results/raw/move_class_b_seed0_check.log`, `results/raw/move_class_status_b_seed0_check.json`: the B seed 0 check run, launched alone before the others
