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

**Ollama model-load degradation has now cost two runs (arm E seed 4 on 2026-09-19, arm F seed 0 on 2026-09-28). Restarting `ollama serve` did NOT fix it on 2026-09-28. Check load time before every LLM launch.**

Symptom both times: `httpcore.ReadTimeout` / `httpx.ReadTimeout` out of an operator call, killing the run mid-GA. It is NOT a dead server and NOT a code fault — `ollama serve` stays alive and the model finishes loading *after* the client has given up. The client timeout is `HPGA_LLM_TIMEOUT_S`, default **120 s** (`hpga/operators.py:114`, applied at `:191`).

Measured on 2026-09-28, same machine, same model (`gemma4:12b`, 6.9 GB blob), same day:

| when | node | unload/reload wall | `load_duration` |
|---|---|---|---|
| 15:00, after a fresh `ollama serve` | 2 users | 7.2 / 8.3 s | 7.0 / 8.0 s |
| 16:30, after ~14 generations of ESMFold swaps | 4 users | 285 / 321 / 297 s | 195 / 225 / 221 s |
| 17:47–18:11, after a fresh `ollama serve` restart | 4 users | 356 (first load) / 351 / 351 / 348 s | 255 / 255 / 255 / 252 s |

A ~40x degradation. Ruled out as causes: disk (`dd` on the blob gave **3.6 GB/s**, it is page-cached), free memory (28 GB free, 27 GB cache), and GPU contention (`nvidia-smi` showed only this user's `llama-server`). What did change is the node filling up with other users' EDA jobs — 2 users at 15:00, 4 at 16:35, 6 by 17:46 (`aryllp` running `dve.exe`/`simv`, `wblanken` running `virtuoso`/`vds`). The load path is CPU-side, so this correlation is the leading explanation, but it is a correlation on two data points, not a proven cause. **We do not control this node's other tenants.**

**The restart did not help this time.** Four loads after a fresh restart took 348–356 s wall (252–255 s `load_duration`), slightly *worse* than the 285–321 s before the restart. That contradicts the 2026-09-19 precedent below, where a restart brought loads back to 13.8 s. So "restart ollama" is no longer a reliable fix. Whatever degrades the load survives a server restart, which fits the node-contention explanation rather than a state leak inside `ollama serve`.

Arm F is more exposed than arms C–E. It swaps ESMFold and Ollama every generation, ~39 loads per seed, so a slow load is paid ~39 times rather than a few. At the measured ~350 s per load that is ~3.8 h of pure model loading per seed, against the ~1 h/seed the design assumes, and each of those loads is a chance to exceed the client timeout.

**Keeping the model resident is not an option on this card.** gemma4:12b takes ~7.9 GiB of GPU memory when loaded: 7,024 MiB weights + 544 MiB KV cache + 127 MiB compute buffer + ~354 MiB vision projector, per Ollama's load log of 2026-09-28. ESMFold takes ~13.8 GB (the figure used in `ESM2_LIKELIHOOD_SCREEN.md`). The Tesla T4 has 15,360 MiB. The two do not fit together, which is why the arms swap them in the first place.

Before any LLM launch:
1. a timed unload/reload measured at **under ~20 s** (restarting `ollama serve` first is fine, but on its own it does not prove anything), and
2. if it is slow, do not launch and do not count on a restart to fix it; the run will either time out or cost 10x.
3. Consider `HPGA_LLM_TIMEOUT_S=600` as margin regardless. It changes no measurement (load time already lands in `llm_s`), it only stops the client giving up on a load that would have finished.

The 2026-09-19 precedent is in `results/SEQUENCE_GA_REPORT.md:823-825`: restarting `ollama serve` brought the load back to 13.8 s. That did not repeat on 2026-09-28 (see the table above). Note that report calls Ollama "a shared service"; on 2026-09-28 it was this user's own process on `localhost:11434`, so restarting it disturbed nobody.

**Pre-existing defect in `experiments/verify_llm_operator_parity.py` (found 2026-09-28, not caused by the change that found it).** In `check` and `active` modes the harness reports **355 of 4410 direct-call cases diverged even when the working tree IS the baseline** — running it in a clean checkout at the same commit gives the identical 355 case IDs and identical coverage counts. So its absolute pass/fail is not trustworthy: a genuinely inert change still shows 355 divergences, and a real regression of that size would be indistinguishable from the floor. `next_generation` specs (13) and the namespace check are clean; only direct calls are affected, and every diverging case is `op=crossover, style=full` — the lattice style gated by `_crossover_sufficiently_mixed`, which reads the module-global `CROSSOVER_MIN_DIFF` that the harness patches per case. The likely cause is that the patch reaches one of the two module objects (baseline loaded from the git blob vs. the imported working tree) and not the other, so the two sides run different mix-gate thresholds. `self` mode is clean (0 diverged), which is why this went unnoticed.
  - **Usable workaround until it is fixed:** run the harness twice, once in a clean checkout at the baseline commit and once in the working tree, and diff the full outputs. Byte-identical output means the change is inert. That is how the arm-F shared edits to `operators.py` were cleared.
  - Fixing it properly means making the harness patch `CROSSOVER_MIN_DIFF` on both module objects (or asserting HEAD-vs-HEAD is clean as a self-test before comparing anything, which would have caught this at the time).

**Other notes:**
- **Environment:**
  - Six `<defunct>` bash zombies are listed under this user. Five have parent PID 3680190 and one (3612654) has parent PID 3515874; both parents are older, still-open `claude` sessions.
  - They are harmless and hold no resources.
  - The five with parent 3680190 are the shells killed in the first half of the session. The sixth predates this session.
- **Arm F seed 0 failed attempt (2026-09-28 15:34–16:17).** Its partial JSON, error file, log and two LLM call logs were moved out of `results/raw/`; they are not results. `results/raw/fitness_prompt_status.json` from that attempt is left untracked and will be overwritten by the next launch. A stuck waiter loop from that session (pid 276722, a `pgrep -f keep_alive` loop that matched its own command line) was killed; it remains a zombie under its dead parent.
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

### To resume arm F (seeds 0–2) when the node is quiet

**1. Measure the reload.** Accept only if every `wall` is **under ~20 s**:
```
cd /scratch/pcanaste/projeto/phase2_agent_memory && /scratch/pcanaste/venv/bin/python -u - <<'EOF'
import time, ollama
c = ollama.Client(host="http://localhost:11434", timeout=900)
for i in (1, 2, 3):
    c.generate(model="gemma4:12b", prompt="", keep_alive=0)
    time.sleep(8)
    t0 = time.perf_counter()
    r = c.generate(model="gemma4:12b", prompt="ok", options={"num_predict": 1})
    print(f"reload #{i}: wall={time.perf_counter()-t0:7.2f}s  load_duration={r.get('load_duration',0)/1e9:7.2f}s")
EOF
```
If `ollama serve` is not running, start it first with the same `PATH`, `LD_LIBRARY_PATH` and `OLLAMA_MODELS` as below.

**2.** Confirm `nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader` shows nothing but Ollama's own runner.

**3. Launch.** The driver skips finished runs, and `HPGA_LLM_TIMEOUT_S=600` adds margin without changing any measurement:
```
cd /scratch/pcanaste/projeto/phase2_agent_memory && export XDG_CACHE_HOME=/scratch/pcanaste/cache HF_HOME=/scratch/pcanaste/cache/huggingface TORCH_HOME=/scratch/pcanaste/cache/torch HF_HUB_OFFLINE=1 PATH=/scratch/pcanaste/ollama/bin:$PATH LD_LIBRARY_PATH=/scratch/pcanaste/ollama/lib:$LD_LIBRARY_PATH OLLAMA_MODELS=/scratch/pcanaste/ollama-models PYTHONHASHSEED=0 HPGA_LLM_TIMEOUT_S=600 && setsid nohup /scratch/pcanaste/venv/bin/python experiments/run_fitness_prompt_ga.py run --arms F --seeds 0 1 2 > results/raw/fitness_prompt_run_F_seeds012.log 2>&1 < /dev/null &
```
Progress goes to `results/raw/fitness_prompt_status.json` and to the log, one line per generation.

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
| `e85eab3` | Fitness-aware operator prompts (arm F): new module, optional kwargs, driver, verifier | yes |
| `f99ffd5` | HANDOFF_MOVE_CLASS.md: record the verify_llm_operator_parity.py check-mode defect | yes |
| (this update) | Ollama load degradation survives a restart; arm F seed 0 attempt cleaned up; resume steps | see `git log` |
