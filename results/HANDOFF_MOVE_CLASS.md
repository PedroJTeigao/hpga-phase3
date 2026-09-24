# Handoff: move-class session (2026-09-23 to 2026-09-24)

Every number below was re-read from a file in this repo, or from a command run at handoff time; the source is named alongside it. Nothing is still running.

## 1. Goals

Evidence 4 of `results/OPERATOR_DOES_NOT_MATTER.md` showed that the greedy oracle's 8 single-position candidates rarely beat their base and gained little when they did. That is a statement about the move class, not about who chooses the move. This session tested whether a different class of move pays more per evaluation. Four deterministic move classes (S1 single substitution, S2 2–5 substitutions, S3 segment replacement, S4 indels) were run inside an otherwise identical GA with no LLM anywhere. The session measured best fitness at a common evaluation budget and, per move, the share that improve on their base and the mean gain when they do.

The second half of the session (2026-09-24) did three things:
- It checked `results/MOVE_CLASS.md` against the raw files and corrected its claims.
- It corrected one fitness range in `results/OPERATOR_DOES_NOT_MATTER.md`.
- It folded the result into `results/PROJECT_SUMMARY.md`, then fast-forwarded `main` to `move-class`.

## 2. Current state

- **Repository:** `/scratch/pcanaste/projeto/phase2_agent_memory`, branch `move-class`. The `phase2` worktree is on `main`.
- **Where the next session works: `/scratch/pcanaste/projeto/phase2_agent_memory`, on `move-class`. Do not work in `/scratch/pcanaste/projeto/phase2`.**
  - `phase2` is the `main` worktree. Experiments here are done on a branch and merged into `main` afterwards, as `move-class` was with `62c3fd5`, so new work should not be committed directly on `main`.
  - Both branches are at `62c3fd5` and both trees are clean, so nothing on disk favours `phase2`. But a Claude session in this project starts in `phase2` by default.
  - That default is what went wrong earlier: the `move-class` branch was first created in the `phase2` worktree and had to be moved here by switching `phase2` back to `main`.
  - Before any edit or launch, `cd` to `phase2_agent_memory` (or use `git -C` with that path) and confirm `git branch --show-current` prints `move-class`.
  - The next-step commands in §6 already `cd` there.
- **HEAD before this handoff update:** `62c3fd508ba51a77a8e6787175c7ac7c3c023b1d` on both `move-class` and `main`.
  - Both are pushed: `origin/move-class` and `origin/main` resolve to the same hash.
  - `main` was fast-forwarded from `76a5fe4` to `62c3fd5`, so the move-class work is merged.
- **Working tree:** clean in both worktrees before this file was edited (`git status --short` returned 0 lines).
- **Driver:** finished, not running.
  - `results/raw/move_class_status.json`: plan 15, done 14, failed 0, skipped 1, `finished: true`, exit reason "all planned runs attempted". The skip is B seed 0, "result file already complete"; it had already been run as the reproduction check.
  - That driver started 2026-09-23T15:27:27-0400 and ended 19:52:47-0400. The last line of `move_class_run.log` is "moveclass finished: all planned runs attempted".
  - The B seed 0 check run is recorded separately in `results/raw/move_class_status_b_seed0_check.json`: plan 1, done 1, failed 0, 15:11:26 to 15:27:04.
- **Detached driver process:** not alive. `ps -p 1829187` returns no process, and `pgrep -af "[r]un_move_class"` matches nothing.
- **Errors:** none.
  - There are no `move_class_*error*` or `*partial*` files.
  - `grep -cE "crashed|Traceback"` returns 0 on both driver logs.
  - All 15 runs completed on attempt 1.
- **GPU:** `nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader` printed nothing at handoff time.
  - `ollama serve` (PID 1009899) is still running but holds no compute process.

**The 15 finished runs.**
- **Best at end:** `summary.final_best` in `results/raw/move_class_<arm>_seed<n>.json`.
- **Best at n_cut:** read from each run's best-so-far curve at n_cut = 278 / 276 / 280 for seeds 0 / 1 / 2.
- **Inert:** both inertness checks passed (`summary.inertness`).
- **Reproduces B:** `summary.reference_match.identical`; applies to the B runs only.

All of these were recomputed from the raw files this session.

| arm | seed | best at end | best at n_cut | distinct folds | inert | reproduces sequence_ga_cmp_B |
|---|---|---|---|---|---|---|
| S1 | 0 | 0.40311 | 0.4031 | 282 | yes | n/a |
| S1 | 1 | 0.38754 | 0.3875 | 282 | yes | n/a |
| S1 | 2 | 0.48156 | 0.4816 | 282 | yes | n/a |
| S2 | 0 | 0.41067 | 0.4107 | 282 | yes | n/a |
| S2 | 1 | 0.6522 | 0.6522 | 282 | yes | n/a |
| S2 | 2 | 0.41749 | 0.4175 | 282 | yes | n/a |
| S3 | 0 | 0.39103 | 0.3910 | 282 | yes | n/a |
| S3 | 1 | 0.41931 | 0.4193 | 282 | yes | n/a |
| S3 | 2 | 0.52653 | 0.5265 | 282 | yes | n/a |
| S4 | 0 | 0.43793 | 0.4379 | 281 | yes | n/a |
| S4 | 1 | 0.49591 | 0.4770 | 282 | yes | n/a |
| S4 | 2 | 0.47501 | 0.4750 | 282 | yes | n/a |
| B | 0 | 0.42026 | 0.4203 | 278 | yes | yes |
| B | 1 | 0.50254 | 0.5025 | 276 | yes | yes |
| B | 2 | 0.50402 | 0.5040 | 280 | yes | yes |

**Headline numbers** (`results/MOVE_CLASS.md`, every figure recomputed from the raw files this session):

- **Significance:** 3 seeds, so the smallest two-sided sign-test p is 0.25. Nothing below is significant on its own.
- **Final fitness:** the only pair ordered in every seed at n_cut is B > S1.
  - S2 seed 1's 0.6522 is the highest single run in the project. It is 0.015 above `sequence_ga_cmp_D_seed1.json` (0.6372) and does not stand apart from earlier runs.
- **Share of moves that improve, pooled:** S1 245/798 (30.7%), S2 273/798 (34.2%), S3 226/798 (28.3%), S4 304/798 (38.1%), B 250/798 (31.3%).
- **Mean gain when a move improves, pooled:** S1 0.0220, S2 0.0343, S3 0.0326, S4 0.0310, B 0.0302.
- **Orderings that hold in every seed, over the whole run:**
  - On share improved, S4 is higher than S1, S2, S3 and B, and S2 is higher than S1, S3 and B.
  - On gain when improved, S2, S3 and S4 are each higher than S1.
- **Qualifications from the per-seed thirds:**
  - S4 is above B in 8 of 9 seed-by-third cells; the exception is early seed 1 (B 40/84, S4 34/84).
  - S4 is strictly highest of all five arms in only 5 of 9 cells, and tied with S2 in a sixth (middle seed 2, 29/84 each).
  - S2 is not above B in the late third in any seed: tied 29/98, then 27/98 against 28/98, then 30/98 against 32/98.
- **Expected gain per move** (hit rate × gain when improved): pooled, S1 is lowest (0.0067 against 0.0092–0.0118). It is not lowest in seed 2, where S3 is lower (0.0074 against 0.0078).
- **Measurement cost:** 2,493 measurement folds, which is 1.74 fold-hours against 2.90 for the GA's own folds.
  - That is 37.5% of fold time, or 37.3% of total driver time.
  - These are wall-clock fold times; nothing in the files records GPU utilisation.

## 3. Files touched

`git diff --name-status 76a5fe4 HEAD` gives 44 paths: 42 with status `A` (new) and 2 with status `M` (modified).

**New code:**
- `hpga/move_class.py`: the move functions B (existing per-site operator, called unchanged), S1, S2, S3 and S4.
- `experiments/run_move_class.py`: the driver.
  - It mirrors `run_sequence_ga_comparison.run_ga`'s deterministic loop and swaps only the model's `deterministic_mutate`.
  - It folds base fitnesses off-budget and asserts inertness.
  - It checks B against the published runs.
- `experiments/summarize_move_class.py`: builds the tables, the summary JSON and the plot from the raw files.

**New results:**
- `results/MOVE_CLASS.md`: the report. Written in `fc2786c`, corrected in `f63efd7` and `cdd8173` (see §5).
- `results/MOVE_CLASS_TABLES.md`: the generated tables.
- `results/move_class_summary.json`: the generated numbers (n_cut, bests, trajectories, payoff, thirds, S4 lengths, measurement cost).
- `results/move_class.png`: best-so-far vs distinct evaluations, mean over seeds.
- `results/HANDOFF_MOVE_CLASS.md`: this file.
- The three generated files are unchanged since `fc2786c`. The later corrections were to the report's prose, so nothing needed regenerating.

**New raw data:**
- `results/raw/move_class_{S1,S2,S3,S4,B}_seed{0,1,2}.json` (15 files): the GA runs, in the `sequence_ga_cmp_*` schema plus per-generation `lengths`.
- `results/raw/move_class_bases_{S1,S2,S3,S4,B}_seed{0,1,2}.json` (15 files): every move with its base and child genomes and fitnesses. Measurement only, kept out of the run files.
- `results/raw/move_class_run.log` and `results/raw/move_class_status.json`: console log and status of the 14-run driver.
- `results/raw/move_class_b_seed0_check.log` and `results/raw/move_class_status_b_seed0_check.json`: console log and status of the B seed 0 reproduction check. The status file was copied before the 14-run driver overwrote `move_class_status.json`.

**Edited files:**
- `results/OPERATOR_DOES_NOT_MATTER.md`: line 77 only. The range "final best fitnesses … sit between 0.38 and 0.60" became best fitness at the common evaluation count across Evidence 1–3, 0.3587 (`sequence_ga_cmp_D_seed0.json`) to 0.6372 (`sequence_ga_cmp_D_seed1.json`).
  - The old range was exactly the Evidence 2–3 arms' range (0.3781–0.5977), so it left out arm D.
- `results/PROJECT_SUMMARY.md`:
  - new §3.7 and findings 22–25
  - the line 103 best-run statement, scoped to the five-arm comparison
  - §6 Seeds and One move class for the oracle
  - §7 item 8
  - §8 Reproducibility
  - one abstract sentence, the title, and the legend entry for MC

**Protected files, confirmed unmodified at handoff:**
```
$ git diff --name-only 76a5fe4 HEAD -- hpga/island.py hpga/worker.py hpga/instrumentation.py | wc -l
0
$ git status --short -- hpga/island.py hpga/worker.py hpga/instrumentation.py | wc -l
0
$ git diff 76a5fe4 HEAD -- results/PHASE3_RESULTS.md | wc -l
0
```
(`76a5fe4` is the commit the branch was created from.)

## 4. What changed (design decisions)

- **S1 was redefined as exactly one substitution per child. The per-site p=0.05 operator is arm B, a reference row.**
  - Why: the existing deterministic baseline is per-site substitution (about 3 substitutions per child), so "S1 = single-position" and "S1 reproduces the baseline" could not both hold.
  - You chose to make S1–S4 all one move per child so they compare as move classes, and to keep B as a fifth row. Single-position gets its own arm because Evidence 4 concerned the oracle's single-position candidate set, not arm B.
- **B re-run for seeds 0, 1 and 2 (15 runs in total, not 12).**
  - Why: the published B files carry no per-move records. Re-running B through the new driver gives it move-payoff numbers, and doubles as an exact reproduction check in all three seeds.
- **Every non-B move class ignores the mutation rate and applies exactly one move to each of the 14 non-elite children per breeding step.**
  - Why: this compares one move against one move.
- **S3 and S4 placement conventions.** S3 draws its span length first, then a start where the whole span fits. S4 redraws direction, span and position when a draw would leave [30, 80] (9 of 798 moves).
  - Why: this keeps the span-length distributions exactly as specified and the genomes within the model's length bounds.
- **Off-budget measurement folds for base fitness.** The base is the post-crossover child, which the GA never folds unless it is an unchanged parent. Bases are folded after the GA finishes, in a separate cache.
  - Why: this gives payoff numbers without changing the search or its budget.
- **The three inertness requirements you set, and how each was met:**
  1. **The run's count and curve are byte-identical to a run without measurement.** A SHA-256 digest of the distinct count, curve, curve tags, populations and fitnesses is compared before and after measurement. A second check replays the GA with move recording off, serving fitness only from the run's own cache; any unseen genome raises. For B, the match against `sequence_ga_cmp_B_seed{0,1,2}.json` is an external third check.
  2. **Measurement folds live in a separate file**, `results/raw/move_class_bases_<arm>_seed<n>.json`, never the run file.
  3. **Measurement cost is reported separately**, as share of fold time and share of total run time, per run and per arm, in `results/MOVE_CLASS.md`.
- **Verification before launch:** B seed 0 was run alone, checked, and committed as `a594dfc` before the other 14 runs were launched.
- **Reporting rules applied in the second half, at your request:**
  - Every share is given with its count.
  - Cost is given as fold time, never GPU time.
  - An arm "beats" another only if it is higher in every seed.
  - The 0.25 sign-test floor is stated inside the conclusion, not only in a banner.
  - The hit-rate-vs-fitness reading is labelled as interpretation, with the alternative reading beside it: the hit-rate differences may be too small to show in three seeds.

## 5. What failed or is unresolved

No run failed. All 15 completed on the first attempt, every inertness assertion held, and all three B runs reproduced the published arm-B files exactly.

**Review of `MOVE_CLASS.md` against the raw files (2026-09-24).**

The check used a script independent of `summarize_move_class.py`, which recomputed each move's gain from its base and child fitness. The stored `gain` and `improved` fields agree in all 3,990 moves.

No number in the report disagreed with the raw files. These claims were corrected:
- **S4 "highest improvement rate in every seed and every third":** true per seed over the whole run and pooled per third, but not per seed per third. Now stated as 8 of 9 cells against B, and strictly highest in 5 of 9 (tied in a sixth).
- **S2's rising gain when improved (0.030 → 0.034 → 0.040):** now stated as pooled and driven by seed 1 (late 0.0697). Seed 0 falls and seed 2 rises modestly.
- **S2 vs B in the late third:** S2 is not above B in any seed. Added to the report.
- **S1 needing the fewest extra folds (397):** the stated reason did not hold, because unchanged-parent bases are cached in every arm. It was replaced with the fact: 379 of 798 bases already cached, against 224–316 for the other arms, and nothing in the files explains why. **This is still unexplained.**
- **"Weakest move class on per-move payoff":** now names the metric (expected gain per move) and says it is pooled, with seed 2 as the exception.
- **"GPU time", "GPU-hours":** now fold time and fold-hours, with a per-arm table giving both share of fold time and share of total run time.
- **Removed:** a sentence comparing per-move expected gain with the spread of final fitness across all runs. It mixed a per-move quantity with a final-fitness quantity.
- **Also:** counts added to every share, a per-seed thirds table, "shows" softened to "is consistent with", and "S1 lowest curve from about 100 evaluations" corrected to "from evaluation 70".

**Left as is, by decision:**
- `results/SEQUENCE_GA_REPORT.md:437` says "D has the single best result of the whole study (seed 1, 0.6372)". That is true within that report's study.
- The project-wide statement in `PROJECT_SUMMARY.md` was updated instead.

**Open design caveat, stated in the report:** move bases are not matched across arms, since each arm mutates its own population. A matched-base test was not run.

**Other notes:**
- **Environment:**
  - Six `<defunct>` bash zombies are listed under this user. Five have parent PID 3680190 and one (3612654) has parent PID 3515874; both parents are older, still-open `claude` sessions.
  - They are harmless and hold no resources.
  - The five with parent 3680190 are the shells killed in the first half of the session. The sixth predates this session.
- **GPU exclusivity during the runs** was checked with `nvidia-smi` before both launches. That output is in the session transcript, not a file. `grep -cE "GPU held|Ollama runner still"` returns 0 on both driver logs.

## 6. Next steps

The merge into `main` is done, so step 1 of the previous handoff is complete. What remains, all optional:

1. **To pin down the per-move ordering, run a matched-base payoff test.**
   - Take a fixed set of post-crossover bases, for example all bases in `results/raw/move_class_bases_B_seed{0,1,2}.json`.
   - Apply every move class to the same bases off-budget.
   - Compare share improved and gain when improved on identical inputs.
   - This needs a new script; nothing for it exists yet.
2. **If final fitness is the question, add seeds to S2, S4 and B.**
   - At 3 seeds the two-sided sign-test floor is 0.25; at 5 seeds it is 0.0625, and 6 or more seeds are needed to go below 0.05.
   - The driver already supports this and skips finished runs:
   ```
   cd /scratch/pcanaste/projeto/phase2_agent_memory && PYTHONHASHSEED=0 setsid nohup /scratch/pcanaste/venv/bin/python experiments/run_move_class.py run --arms S2 S4 B --seeds 3 4 5 > results/raw/move_class_run_seeds345.log 2>&1 < /dev/null &
   /scratch/pcanaste/venv/bin/python experiments/summarize_move_class.py --seeds 0 1 2 3 4 5
   ```
   - The summarizer expects all five arms for every listed seed. Either run S1 and S3 on the new seeds as well, or change its `ARMS` tuple first.
   - B seeds 3 and 4 will be checked against `sequence_ga_cmp_B_seed{3,4}.json` automatically.
   - Any new seeds also mean updating `results/MOVE_CLASS.md`, `PROJECT_SUMMARY.md` §3.7 and findings 22–25.
3. **Before any launch:** confirm `nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader` is empty, and use `/scratch/pcanaste/venv/bin/python`.

## Commits this session

All on `move-class`. `main` was fast-forwarded from `76a5fe4` to `62c3fd5`.

| hash | message | pushed |
|---|---|---|
| `a594dfc` | Move class: operators, driver, summarizer; B seed 0 reproduces sequence_ga_cmp_B_seed0 exactly | yes |
| `fc2786c` | Move class: 15 runs (S1-S4 + B reference, seeds 0-2), raw logs and MOVE_CLASS.md | yes |
| `531b270` | Handoff for the move-class session | yes |
| `f63efd7` | MOVE_CLASS.md: corrections — counts, per-seed tables, fold time not GPU time, sign-test floor in conclusion | yes |
| `cdd8173` | MOVE_CLASS.md conclusion and §1 fixes; OPERATOR_DOES_NOT_MATTER.md: correct fitness range to 0.3587–0.6372 | yes |
| `62c3fd5` | PROJECT_SUMMARY.md: §3.7 move class, findings 22–25, limits/questions/reproducibility/abstract updates | yes, on `move-class` and `main` |
| (this update) | not yet committed | no |
