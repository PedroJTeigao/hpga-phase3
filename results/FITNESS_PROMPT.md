# Fitness in the prompt: does telling the LLM operator the genome's fitness change what it does?

**Direction-finding, not a result.** Three seeds. Under the fixed rule used throughout this project (X beats Y only if X is higher in *every* seed), the smallest two-sided sign-test p reachable with 3 seeds is 2/2³ = **0.25**. Nothing in this document is significant on its own under that rule. The one comparison below that uses a test on pooled counts (the invalid-output rate) says so where it does.

## The question, and why it is asked

Published positive results for LLMs as evolutionary or optimisation operators put the objective value in the prompt. OPRO (Yang et al., 2023, "Large Language Models as Optimizers"), for example, shows the model earlier solutions together with their scores. This project's operator prompts never did. Arms C, D and E of `results/SEQUENCE_GA_REPORT.md` showed the model sequences and asked for a mutation or a crossover, but never the fitness of any sequence. So a finding here that the LLM operator adds nothing (`results/OPERATOR_DOES_NOT_MATTER.md`) could not rule out the simplest explanation: the model was never told which sequences were good.

Arm F asks that question directly. It is arm C with one difference: each operator prompt carries a labelled line giving fitness. Everything else — prompt text, prompt styles, model, temperature, GA and budget — is arm C.

## Design

Everything is arm C of `experiments/run_sequence_ga_comparison.py`, imported rather than restated (`experiments/run_fitness_prompt_ga.py` imports it as `base`):

- population 16, 20 evaluated populations (generation 0 = random start, so 19 breeding steps), crossover 0.9, tournament 3, elitism 2, length bounds [30, 80]
- `Random(seed)` drives everything, and every arm starts from the same generation-0 population per seed. F's generation-0 population and fitnesses are identical to C's in all three seeds.
- fitness = TM-score of the ESMFold fold against 7UR7 chain A, through the same cache-aware evaluator
- operator prompts: mutate style `position`, crossover style `segment`, `gemma4:12b`, temperature 0.7, default retries (2)
- seeds 0, 1, 2

**The fitness line.** Crossover's two inputs are population members, so each has an exact fitness, shown next to its sequence (`Parent 1 fitness: 0.25744`). Mutate's input is a post-crossover child that the GA has not folded, so its own fitness is unknown at mutate time. Folding it would change the budget. So mutate has two cases, each with its own label, and the case is recorded per call as `fitness_basis`:

| basis | when | line shown |
|---|---|---|
| **own** | the child is byte-identical to one of its parents, so its fitness is that parent's, exactly | `Fitness of this sequence: 0.32551` |
| **inherited** | the child differs from both parents | `Parent fitness scores: 0.25744 and 0.29143` |

Both cases always show a line, so whether a line appears is never tied to whether the child is a copy. Values are shown to 5 decimals.

**How the line is added.** `hpga/sequence_model_fitness.py` does not copy the prompt text. It takes the plan that `hpga/sequence_model.py` builds (arm C's plan, unmodified) and inserts one line after an anchor line it asserts is present. Only `build_prompt` is replaced. The parser, system prompt, retry hint and token limit are C's own objects. `hpga/sequence_model.py` is untouched.

## Verification: the change is inert until a fitness context is passed

- **Byte-identity of prompts.** `experiments/verify_fitness_prompts.py check`: 672 cases, 0 failed. The cases cover both operators, both mutate cases, four length pairs, four fitness pairs and the retry hint. Each case asserts four things:
  - removing the fitness line gives the base prompt byte for byte;
  - with `fitness=None`, the plan equals the base plan in every field, including parse results on eight probe responses;
  - an anchor line that has moved raises;
  - an out-of-scope style raises.

  Re-run at `0d05721`: `RESULT: PASS`.
- **Arm B reproduces the published baseline exactly, seed 0.** Arm B was run through this driver and matched `results/raw/sequence_ga_cmp_B_seed0.json` on:
  - distinct-fold count;
  - the best-so-far curve and its generation tags;
  - all 20 generations' populations and fitnesses (`reference_match.identical = true`).

  This covers the shared edits to `hpga/operators.py`: the optional `fitness` keyword arguments and `next_generation`'s pass-through. It ran from the working tree committed as `e85eab3`, on 2026-09-28. Seeds 1 and 2 of B were not re-run. The check's run file was written to a session scratch directory, not to `results/raw/`, so it is not in the repository. The commit message of `e85eab3` records the result.
- **The shared operator edits change nothing for any other arm.** `experiments/verify_llm_operator_parity.py` gave byte-identical output from this tree and from a clean checkout at the parent commit, in both `check` and `active` mode. That harness has a known pre-existing defect: 355 cases diverge even when the tree is the baseline (`results/HANDOFF_MOVE_CLASS.md`, section 5). That is why the comparison was tree-vs-clean-checkout output, not the harness's own pass/fail.
- **Measurement is inert, checked twice per run, exactly as in `MOVE_CLASS.md`.** Mutate bases were folded only after the GA had finished, into a separate cache and file (`fitness_prompt_bases_F_seed<n>.json`). Two checks, both passed in all three runs (`summary.inertness`):
  - a SHA-256 digest of the GA's outputs is identical before and after measurement;
  - a replay from the recorded operator outputs is identical to the run.

  Each run swaps ESMFold and Ollama 39 times; all 4 swap integrity checks per run (a fitness refolded before and after a swap) gave identical values.
- **The fitness line reached the model as designed.** From the call logs and bases files:
  - all 266 mutate and 133 crossover calls per seed are recorded with a fitness line shown. Every mutate attempt sent to the model carried the correct label, retries included;
  - every value recorded as shown appears in its prompt text, for both operators;
  - every shown value is a fitness that occurs in an evaluated population;
  - every own-basis value equals the base's measured fitness (178 of 178).

**What measuring cost.** 569 extra folds over the three seeds (173 / 206 / 190), which is 38.7% of all fold time (36.9% / 40.1% / 38.8% per seed). As a share of each run's end-to-end driver time it is only 11.3% / 12.6% / 12.2%, because LLM time dominates these runs.

**Launch history.** The first attempt at seed 0 (2026-09-28 15:34) failed after generation 14 with `httpx.ReadTimeout`: loading the Ollama model had slowed to ~350 s on the shared node. It was discarded and is not in `results/raw/`; see `results/HANDOFF_MOVE_CLASS.md`. The recorded runs were launched at 19:04 after three model reloads measured 7–15 s. They used `HPGA_LLM_TIMEOUT_S=600` instead of the default 120 s. The timeout only sets how long the client waits before giving up, and load time is already counted in `llm_s`. No run hit it, and every run completed on attempt 1.

## 1. Best fitness at the common evaluation count

Each seed is read at `n_cut`, the smallest distinct-fold count among F, C and B in that seed: 278, 276 and 280 for seeds 0, 1 and 2. B sets `n_cut` in every seed, as in `MOVE_CLASS.md`.

| arm | seed 0 | seed 1 | seed 2 | mean |
|---|---|---|---|---|
| F fitness in prompt | 0.3739 | 0.4116 | **0.5281** | 0.4379 |
| C same prompts, no fitness | **0.4632** | 0.4745 | 0.5138 | 0.4838 |
| B deterministic | 0.4203 | **0.5025** | 0.5040 | 0.4756 |

**No ordering holds in every seed, in either direction.**

- F is above C in seed 2 only, and below C in seeds 0 and 1.
- F is above B in seed 2 only, and below B in seeds 0 and 1.
- C and B are not ordered in every seed either: C is higher in seeds 0 and 2, B in seed 1.

With three seeds the smallest two-sided sign-test p is 0.25, so even a 3-of-3 ordering would not be significant. The result is **no demonstrated difference** in best fitness between showing the fitness and not showing it. It is not evidence that no difference exists. At full budgets F ends at 0.3739 / 0.4116 / 0.5281 (281 / 282 / 282 folds), which does not change any comparison.

## 2. Output compliance: the model repeats the current letter more often

This is the new finding. It is a **change in the operator's behaviour**, not a change in the search outcome (section 1).

Invalid-output rate is failed attempts ÷ all attempts. Requests per call is attempts ÷ accepted LLM calls. From the per-attempt call logs and `summary.operators`:

| | seed 0 | seed 1 | seed 2 | pooled |
|---|---|---|---|---|
| **F mutate**, invalid attempts | 38/304 (12.5%) | 30/296 (10.1%) | 35/300 (11.7%) | **103/900 (11.4%)** |
| **C mutate**, invalid attempts | 9/275 (3.3%) | 3/269 (1.1%) | 28/294 (9.5%) | **40/838 (4.8%)** |
| F mutate, requests per call | 1.143 | 1.113 | 1.128 | |
| C mutate, requests per call | 1.034 | 1.011 | 1.105 | |
| F crossover, invalid attempts | 0/122 | 0/126 | 0/117 | 0/365 |
| C crossover, invalid attempts | 0/124 | 0/124 | 0/117 | 0/365 |

- **F is higher than C in each of the three seeds.** Seed 2 is close: 11.7% against 9.5%. On the fixed rule that is 3 of 3, floor 0.25.
- **C's two further seeds do not overlap F's range.** C seeds 3 and 4 are 18/283 (6.4%) and 15/281 (5.3%). Over all five C seeds the pooled rate is 73/1402 (5.2%), and C's highest seed (9.5%) is below F's lowest (10.1%).
- **Pooled attempts give Fisher's exact p ≈ 3×10⁻⁷** (F seeds 0–2 against C seeds 0–2; 8×10⁻⁸ against all five C seeds). Retries of the same call are not independent attempts, so this p is indicative, not exact.
- **Crossover is unaffected.** Neither arm produced an invalid crossover attempt, and requests per call is 1.000 in every seed of both.
- **Every failure is the same failure: 103 of 103 invalid F attempts named a NEW letter equal to the letter already at that position.** No failure was a format error, a wrong number of lines, a position out of range or a repeated position. C's failures were all the same kind as well: 73 of 73 over five seeds. Both prompts state the rule in so many words: "the new letter MUST differ from whatever letter is currently at that position (copying the current letter back is not a change)". So the model followed the output format in every attempt. What rose was how often it proposed a "change" that changes nothing.
- **Retries recover less often in F.**

  | | first attempts failed | failed again on attempt 2 | failed on attempt 3 | fell back to deterministic |
  |---|---|---|---|---|
  | F | 93 of 798 | 9 of 93 | 1 of 9 | 1 (seed 2) |
  | C, seeds 0–2 | 40 of 798 | 0 of 40 | — | 0 |
- **The rate does not depend on the label.** Own-basis attempts are 23/200 (11.5%) invalid and inherited-basis attempts 80/700 (11.4%).
- F's mutate prompts are one line longer: 330 input tokens per attempt on average, against 309 for C.

Output-format compliance has been the operative mechanism in this project's LLM operators. This result does not bear on that mechanism directly: format compliance did not change. The rule that changed is a content rule stated in the prompt. The cost here was 30–38 retries per run against C's 3–28, plus one fallback in 798 calls. It did not show up in section 1.

## 3. Payoff per move, against the move classes

As in `MOVE_CLASS.md` section 3:
- A move is one mutation call on one post-crossover child. It improves if the child's fitness is strictly above its base's.
- There are 266 moves per run, over all 19 breeding steps.
- The move-class and B figures are recomputed from `move_class_bases_*` and match `MOVE_CLASS.md`.

Arm C has no bases file, so **there is no C row**. The payoff of an LLM chooser without the fitness line was not measured.

| arm | improved: seed 0 / 1 / 2 | **improved, pooled** | gain if improved: seed 0 / 1 / 2 | **gain if improved, pooled** | expected positive gain per move, pooled |
|---|---|---|---|---|---|
| **F** fitness in prompt | 93/266 (35.0%) / 86/266 (32.3%) / 73/266 (27.4%) | **252/798 (31.6%)** | 0.0253 / 0.0341 / 0.0405 | **0.0327** | 0.0103 |
| S1 single substitution | 65 (24.4%) / 89 (33.5%) / 91 (34.2%) | **245/798 (30.7%)** | 0.0230 / 0.0204 / 0.0227 | **0.0220** | 0.0067 |
| S2 2–5 substitutions | 91 (34.2%) / 90 (33.8%) / 92 (34.6%) | **273/798 (34.2%)** | 0.0293 / 0.0487 / 0.0252 | **0.0343** | 0.0118 |
| S3 segment 5–15 | 88 (33.1%) / 84 (31.6%) / 54 (20.3%) | **226/798 (28.3%)** | 0.0296 / 0.0332 / 0.0366 | **0.0326** | 0.0092 |
| S4 indel 1–5 | 92 (34.6%) / 105 (39.5%) / 107 (40.2%) | **304/798 (38.1%)** | 0.0253 / 0.0360 / 0.0311 | **0.0310** | 0.0118 |
| B per-site p=0.05 | 77 (28.9%) / 89 (33.5%) / 84 (31.6%) | **250/798 (31.3%)** | 0.0199 / 0.0339 / 0.0358 | **0.0302** | 0.0095 |

Orderings of F that hold in every seed:

- **Share improved: F is above S3 in every seed.** It is not ordered against S1, S2, S4 or B.
- **Gain if improved: F is above S1 and above B in every seed.** It is not ordered against S2, S3 or S4.

Most of F's moves hurt, as in every arm: 546/798 (68.4%) are worse than their base, against 62–72% for the others. F's mean gain over all moves is −0.0313, against −0.0200 to −0.0383. By thirds of the run (steps 0–5 / 6–11 / 12–18), pooled, F improves 39.7% / 27.0% / 28.6% of the time, with gain if improved 0.0316 / 0.0325 / 0.0342. That is the same early-high, then-lower shape as every arm in `MOVE_CLASS.md` section 4.

**These rows are not matched on move size.** In F the number of positions changed is not the model's choice: it is round(0.05 × length). F changed 3 positions in 762 of 798 moves, 2 in 22 and 4 in 14. The model chooses only which positions and which letters. The nearest move class by size is S2 (2–5 positions, uniform), which F does not beat on either measure in every seed.

## 4. Does the operator's choice follow the fitness it was shown?

Chosen positions and letters are read from the parsed responses of accepted mutate calls in both arms: 797 for F (the one fallback move was not the model's choice), 798 for C.

**Positions differ by label.** Mean relative position of the chosen positions (0 = start, 1 = end), averaged per call:

| group | calls | mean relative position | share of changes in the first third of the sequence |
|---|---|---|---|
| F own | 177 | 0.287 | 343/526 (65%) |
| F inherited | 620 | 0.339 | 1122/1858 (60%) |
| C, base already in a population | 171 | 0.305 | |
| C, base new | 627 | 0.301 | |
| C, all accepted calls | 798 | | 1663/2628 (63%) |

- F own vs F inherited: permutation p = 0.0002, over per-call means, 5,000 permutations.
- **Control:** C has no basis field. So C's calls were split by whether the base genome had already appeared in an evaluated population, which is what makes a base "own" in F. In F that split reproduces the recorded basis in 796 of 797 calls. In C the same split shows no difference: 0.305 against 0.301, p = 0.40, same test. So in F the position difference goes with the fitness line's label, not with the base being a parent copy.
- **All three LLM groups concentrate their changes at the start of the sequence** (60–65% in the first third). That is C's pre-existing behaviour, not something the fitness line introduced.

**Within a label, positions do not follow the value shown.** Rank correlation between the value shown and mean relative position: own ρ = −0.132 (p = 0.08, n = 177); inherited ρ = +0.050 (p = 0.21, n = 620). For inherited, the value used is the mean of the two parent scores.

**Letters.** Own vs inherited letter distributions do not differ detectably: permutation χ², p = 0.09. G is 25% of own changes and 22% of inherited. F's pooled letter distribution differs from C's (p < 0.001), but not in a direction that holds across seeds, so the pooled test is not informative:

| seed | G share, F / C | R share, F / C |
|---|---|---|
| 0 | 0.279 / 0.208 | 0.028 / 0.107 |
| 1 | 0.157 / 0.192 | 0.173 / 0.080 |
| 2 | 0.242 / 0.177 | 0.120 / 0.055 |

**Payoff by label.** Own-basis moves improve 36/178 (20.2%), gain if improved 0.0168. Inherited-basis moves improve 216/620 (34.8%), gain 0.0354. **This gap is not specific to F.** Splitting every deterministic arm the same way (was the base already in the run's cache?) gives a gap of the same direction and similar size, with no fitness in any prompt:

| arm | base already cached: improved, gain if improved | base new: improved, gain if improved | base fitness, cached / new |
|---|---|---|---|
| F | 36/179 (20.1%), 0.0168 | 216/619 (34.9%), 0.0354 | 0.375 / 0.333 |
| S1 | 77/379 (20.3%), 0.0156 | 168/419 (40.1%), 0.0249 | 0.371 / 0.323 |
| S2 | 38/224 (17.0%), 0.0225 | 235/574 (40.9%), 0.0363 | 0.384 / 0.324 |
| S3 | 45/316 (14.2%), 0.0310 | 181/482 (37.6%), 0.0330 | 0.368 / 0.306 |
| S4 | 66/237 (27.8%), 0.0241 | 238/561 (42.4%), 0.0329 | 0.373 / 0.329 |
| B | 48/286 (16.8%), 0.0264 | 202/512 (39.5%), 0.0311 | 0.390 / 0.345 |

In F, "base already cached" is the own-basis set plus one inherited move whose base had been evaluated earlier as a different lineage. Cached bases are fitter in every arm, and a fitter base is harder to improve on. The rank correlation of base fitness with gain is −0.50 to −0.69 in every deterministic arm, and −0.57 in F.

**Payoff by value shown.** Within each label, moves shown a higher value improve less often:

| label | value shown, lower / middle / upper third | improved |
|---|---|---|
| own | 0.215–0.340 / 0.340–0.376 / 0.376–0.528 | 22/59 (37.3%) / 7/59 (11.9%) / 7/60 (11.7%) |
| inherited | 0.226–0.330 / 0.332–0.383 / 0.383–0.523 | 89/206 (43.2%) / 73/207 (35.3%) / 54/207 (26.1%) |

The value shown is not independent of the base:
- for own it *is* the base's fitness (ρ = 1.0);
- for inherited it correlates with base fitness (ρ = +0.63);
- in both labels it correlates with breeding step (ρ ≈ +0.6).

For inherited moves, controlling for base fitness alone (partial rank correlation) leaves a small positive association of value shown with gain: +0.14, p ≈ 0.001. Step was not controlled for. The value shown was never varied independently of the genome, so nothing in this section separates an effect of the number from an effect of the base it describes.

## Scope and limits

- **Three seeds.** Every "in every seed" statement is 3 of 3, with a two-sided sign-test floor of 0.25. The fitness result is no demonstrated difference, not a demonstrated absence of one.
- **Most mutate calls were not shown their own fitness.** 78% of mutate calls (620 of 798) were shown the parents' scores, because a post-crossover child's own fitness is unknown until it is folded. Only 22% saw the fitness of the sequence they were editing. That share varied by seed (29%, 14%, 23%) and is set by the trajectory, not by the design:
  - own calls are exactly the children of crossovers that returned both parents unchanged (39, 19 and 31 such crossovers);
  - most of those are the crossover-rate gate (11, 7, 16) or identical parents (23, 12, 15);
  - seed 0 has 5 more that neither accounts for. These look like recombinations that reproduced a parent, but that was not checked.

  Crossover always saw both parents' exact fitness.
- **The own/inherited payoff gap is explained by base fitness, not by the label.** It replicates in all five deterministic arms, which saw no fitness at all (section 4). It is not evidence that the fitness line changed payoff.
- **Label and value are confounded between the two line formats.** "Fitness of this sequence: x" and "Parent fitness scores: a and b" differ in wording, in the number of values and in what the values refer to. The position difference in section 4 follows the label. These runs cannot say whether the wording, the number of values or the values themselves cause it.
- **The value shown was never randomised.** Every relation between the number shown and what the operator did is observational within one arm. The number is correlated with base fitness and with how far the run has progressed.
- **No payoff for C.** Section 3 compares F with deterministic move classes. The per-move payoff of the same LLM operator without the fitness line was not measured, so section 3 cannot say whether the fitness line changed payoff.
- **Move size is fixed by length, not chosen.** The model chooses positions and letters only, so "what the operator does" can change only in those two respects.
- **B was reproduced for seed 0 only,** and that check's run file is not in `results/raw/`.
- **F and C ran ten days apart.** C ran on 2026-09-18/19 and F on 2026-09-28. Both used the model tag `gemma4:12b` served by a local Ollama. The run files record the tag, not the model blob or the Ollama version, so that the same model build served both is assumed, not verified. That matters most for section 2, which compares the two arms' output behaviour directly.
- **One precision, one prompt style per operator.** Five decimals, mutate `position`, crossover `segment`. Other formats — ranks instead of raw scores, several scored examples, a stated goal — were not tested.
- **Same scope as every other result in this project:** one target (7UR7 chain A, 63 residues), one fitness function, population 16, 20 generations, about 280 folds per run.

## Files

- `hpga/sequence_model_fitness.py`: the fitness-aware prompt plans (inserts one line into `sequence_model.py`'s plan)
- `hpga/operators.py`, `hpga/genome_model.py`, `hpga/config.py`: the optional `fitness` keyword arguments, the `next_generation` pass-through and the `sequence_fitness` genome model
- `experiments/run_fitness_prompt_ga.py`: the driver, including the reference match, the measurement folds and the inertness checks
- `experiments/verify_fitness_prompts.py`: the 672-case byte-identity check
- `results/raw/fitness_prompt_F_seed{0,1,2}.json`: the three runs, in the same schema as `sequence_ga_cmp_*`, plus `fitness_line` and `inertness` in `summary`
- `results/raw/fitness_prompt_bases_F_seed{0,1,2}.json`: every mutate and crossover call, with its fitness line fields, base and child genomes and fitnesses (measurement only)
- `results/raw/llm_operator_calls_fitprompt_F_seed{0,1,2}_*.jsonl`: every LLM attempt, with prompt, response, validity and fitness line fields
- `results/raw/fitness_prompt_run_F_seeds012.log`, `results/raw/fitness_prompt_status.json`: the launch
