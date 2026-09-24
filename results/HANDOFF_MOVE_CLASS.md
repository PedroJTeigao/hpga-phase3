# Handoff: move-class session (2026-09-23)

Every number below was re-read from a file in this repo at handoff time; the source file is named alongside it. Nothing is still running.

## 1. Goals

Evidence 4 of `results/OPERATOR_DOES_NOT_MATTER.md` showed that the greedy oracle's 8 single-position candidates rarely beat their base and gained little when they did. That is a statement about the move class, not about who chooses the move. This session tested whether a different class of move pays more per evaluation. Four deterministic move classes (S1 single substitution, S2 2–5 substitutions, S3 segment replacement, S4 indels) were run inside an otherwise identical GA with no LLM anywhere. The session measured best fitness at a common evaluation budget and, per move, the share that improve on their base and the mean gain when they do.

## 2. Current state

- **Repository:** `/scratch/pcanaste/projeto/phase2_agent_memory`, branch `move-class`.
- **HEAD before this handoff's commit:** `fc2786c3bfaaa7ae8217e44ffb4b21c3a05a5562`. The commit that adds this file sits on top of it; see the commit list at the end.
- **Working tree:** clean before this file was written (`git status --short` returned 0 lines). This file is the only addition.
- **Driver:** finished, not running.
  - `results/raw/move_class_status.json`: plan 15, done 14, failed 0, skipped 1 (B seed 0, "result file already complete", run earlier as the reproduction check), `finished: true`, exit reason "all planned runs attempted".
  - That driver started 2026-09-23T15:27:27-0400 and ended 19:52:47-0400.
  - The B seed 0 check run is recorded separately in `results/raw/move_class_status_b_seed0_check.json`: plan 1, done 1, failed 0, 15:11:26 to 15:27:04.
- **Detached driver process:** not alive. `ps -p 1829187` returns no process, and `pgrep -af "[r]un_move_class"` matches no Python process.
- **Errors:** none. There are no `move_class_*.error.txt` or `*.partial.json` files, and `grep -cE "crashed|Traceback"` returns 0 on both driver logs. All 15 runs completed on attempt 1.

**The 15 finished runs.** "Best at end" is `summary.final_best` in `results/raw/move_class_<arm>_seed<n>.json`. "Best at n_cut" is from `results/move_class_summary.json`, where n_cut = 278 / 276 / 280 for seeds 0 / 1 / 2. "Inert" means both inertness checks passed (`summary.inertness`). "Reproduces B" is `summary.reference_match.identical`, which applies to the B runs only.

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

**Headline numbers** (`results/move_class_summary.json`):

- **Final fitness:** the only pair ordered in every seed at n_cut is B > S1.
- **Share of moves that improve, pooled:** S1 30.7%, S2 34.2%, S3 28.3%, S4 38.1%, B 31.3%.
- **Mean gain when a move improves, pooled:** S1 0.0220, S2 0.0343, S3 0.0326, S4 0.0310, B 0.0302.
- **Orderings that hold in every seed:** on share improved, S4 is higher than S1, S2, S3 and B, and S2 is higher than S1, S3 and B. On gain when improved, S2, S3 and S4 are each higher than S1.
- **Measurement cost:** 2,493 measurement folds, 37.5% of all fold time.

## 3. Files touched

Verified with `git diff --name-status 76a5fe4 HEAD`: 41 paths, all status `A` (new). No existing file was modified.

**New code:**
- `hpga/move_class.py`: the move functions B (existing per-site operator, called unchanged), S1, S2, S3 and S4.
- `experiments/run_move_class.py`: the driver. It mirrors `run_sequence_ga_comparison.run_ga`'s deterministic loop, swaps only the model's `deterministic_mutate`, folds base fitnesses off-budget, asserts inertness, and checks B against the published runs.
- `experiments/summarize_move_class.py`: builds the report tables, the summary JSON and the plot from the raw files.

**New results:**
- `results/MOVE_CLASS.md`: the report (sections 1–5, verification, measurement cost, caveats).
- `results/MOVE_CLASS_TABLES.md`: the generated tables behind the report.
- `results/move_class_summary.json`: the generated numbers (n_cut, bests, trajectories, payoff, thirds, S4 lengths, measurement cost).
- `results/move_class.png`: best-so-far vs distinct evaluations, mean over seeds.

**New raw data:**
- `results/raw/move_class_{S1,S2,S3,S4,B}_seed{0,1,2}.json` (15 files): the GA runs, in the `sequence_ga_cmp_*` schema plus per-generation `lengths`.
- `results/raw/move_class_bases_{S1,S2,S3,S4,B}_seed{0,1,2}.json` (15 files): every move with its base and child genomes and fitnesses. Measurement only, kept out of the run files.
- `results/raw/move_class_run.log`: console log of the 14-run driver.
- `results/raw/move_class_status.json`: status of the 14-run driver.
- `results/raw/move_class_b_seed0_check.log`: console log of the B seed 0 reproduction check.
- `results/raw/move_class_status_b_seed0_check.json`: status of that check, copied before the 14-run driver overwrote `move_class_status.json`.

**Edited files:** none.

**Protected files, confirmed unmodified:**
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
- **S3 and S4 placement conventions.** S3 draws its span length first, then a start where the whole span fits. S4 redraws direction, span and position when a draw would leave [30, 80].
  - Why: this keeps the span-length distributions exactly as specified and the genomes within the model's length bounds.
- **Off-budget measurement folds for base fitness.** The base is the post-crossover child, which the GA never folds unless it is an unchanged parent. Bases are folded after the GA finishes, in a separate cache.
  - Why: this gives payoff numbers without changing the search or its budget.
- **The three inertness requirements you set, and how each was met:**
  1. **The run's count and curve are byte-identical to a run without measurement.** A SHA-256 digest of the distinct count, curve, curve tags, populations and fitnesses is compared before and after measurement. A second check replays the GA with move recording off, serving fitness only from the run's own cache; any unseen genome raises. Why: this proves measuring did not perturb the search. For B, the match against `sequence_ga_cmp_B_seed{0,1,2}.json` is an external third check.
  2. **Measurement folds live in a separate file**, `results/raw/move_class_bases_<arm>_seed<n>.json`, never the run file. Why: downstream summary scripts cannot pick them up by accident.
  3. **Measurement cost is reported separately:** per-run fold counts and share of fold time are in `results/MOVE_CLASS.md`. Why: "did measuring change the result" has a visible answer.
- **Verification before launch:** B seed 0 was run alone and checked, and committed as `a594dfc`, before the other 14 runs were launched.
  - Why: this was your requirement that the flag's baseline setting reproduce the existing run exactly first.

## 5. What failed or is unresolved

Nothing failed. All 15 runs completed on the first attempt, every inertness assertion held, and all three B runs reproduced the published arm-B files exactly. Things that are open or worth knowing:

- **Nothing was skipped from your spec.** The driver's "skipped" entry is B seed 0, which was already complete from the check run.
- **GPU exclusivity.** I checked it with `nvidia-smi` immediately before both launches. That output lives in the session transcript, not in a file. The file-level evidence is that the driver's GPU-wait check never logged a warning: `grep -cE "GPU held|Ollama runner still"` returns 0 on both logs.
- **`ollama serve` was left running.** It held no GPU memory at launch and is not used by this experiment.
- **Session cleanup.** Five leftover shells from earlier sessions were killed (PIDs 369064, 1019577, 4148895, 4164980 and 1536079, plus the stray `find` 1536083). The four watchers never exited on their own, because their `pgrep -f` matched their own command line. Five of them are still listed as `<defunct>` zombies, waiting for their parent (an older, still-open `claude` session, PID 3680190) to reap them. They are harmless and hold no resources.
- **Branch placement.** The `move-class` branch was first created in the `phase2` worktree. It was moved here by switching `phase2` back to `main` (confirmed: `phase2` is on `main`).
- **The main open design caveat, stated in the report:** the move bases are not matched across arms, since each arm mutates its own population. A matched-base test was not run.

## 6. Next steps

`results/MOVE_CLASS.md` is already complete, so there is nothing left to finish there. In order:

1. **Review `results/MOVE_CLASS.md`** and decide whether to merge `move-class` into `main`, as earlier experiments were:
   ```
   cd /scratch/pcanaste/projeto/phase2_agent_memory && git checkout main && git merge --no-ff move-class && git push origin main
   ```
2. **If you want the per-move ordering pinned down, run a matched-base payoff test.** Take a fixed set of post-crossover bases, for example all bases in `results/raw/move_class_bases_B_seed{0,1,2}.json`, apply every move class to the same bases off-budget, and compare share improved and gain when improved on identical inputs. This needs a new script; nothing for it exists yet.
3. **If final fitness is the question, add seeds to S2, S4 and B.** At 3 seeds the sign-test floor is 0.25. At 5 seeds it is 0.0625 (so 6+ seeds are needed to go below 0.05). The driver already supports it, and finished runs are skipped:
   ```
   cd /scratch/pcanaste/projeto/phase2_agent_memory && PYTHONHASHSEED=0 setsid nohup /scratch/pcanaste/venv/bin/python experiments/run_move_class.py run --arms S2 S4 B --seeds 3 4 5 > results/raw/move_class_run_seeds345.log 2>&1 < /dev/null &
   /scratch/pcanaste/venv/bin/python experiments/summarize_move_class.py --seeds 0 1 2 3 4 5
   ```
   The summarizer currently expects all five arms for every listed seed. Either run S1 and S3 on the new seeds as well, or change its `ARMS` tuple first. B seeds 3 and 4 will also be checked against `sequence_ga_cmp_B_seed{3,4}.json` automatically.
4. **Before any launch:** confirm `nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader` is empty, and use `/scratch/pcanaste/venv/bin/python`.

## Commits this session

| hash | message | pushed |
|---|---|---|
| `a594dfc` | Move class: operators, driver, summarizer; B seed 0 reproduces sequence_ga_cmp_B_seed0 exactly | pushed with this handoff (branch had no upstream before) |
| `fc2786c` | Move class: 15 runs (S1-S4 + B reference, seeds 0-2), raw logs and MOVE_CLASS.md | pushed with this handoff |
| (this commit) | Handoff for the move-class session | pushed with this handoff; hash in `git log -1` |
