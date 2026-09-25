# On this problem, at this budget, the operator is not what determines search quality

*A standalone account of one claim, the evidence for it, and its scope. Written to be read without the rest of the repository; source files are named so each number can be checked. No new runs were made for this document: everything below comes from experiments already committed on this branch (`agent-memory`) or on `main`.*

## The claim

Five separate ways of choosing how a genome changes at each breeding step were compared against each other, on the same protein-design task, at the same evaluation budget: two deterministic operators, three LLM-driven schemes that differ in how much of the search's own history they can see, an architecture (circles) that adds coordination and communication, plain random injection with no search reasoning at all, and a greedy hill-climber with direct read access to the real fitness function. **None of the interventions that changed the operator's behaviour changed the search's outcome, and the intervention with the most information available to it — the oracle, the only arm that could evaluate a real candidate before committing to it — was the worst of the five.** Four operator interventions changed operator behaviour exactly as predicted and never moved fitness; random injection matched all of them. Then an operator with direct fitness access still did not beat random injection — it did worse. Read together, the natural reading is that on this problem, at this budget, the *choice of perturbation* is not the thing determining how well the search does; something else (most plausibly selection, operating over a landscape where single-residue edits carry little usable signal at this evaluation count) is doing the work.

Read this as "as far as tested": one 63-residue target, one fitness function (TM-score of an ESMFold-predicted structure against a fixed reference), one move class (single or few-position edits to a fixed-alphabet string), one main model (`gemma4:12b`), population 16, 20 evaluated populations, 3 to 5 seeds depending on the arm. The Scope section below states this precisely; nothing here is a claim about other problems, larger populations, more generations, or larger models.

## Setting common to every arm below

- **Target and fitness.** A 63-residue reference (PDB 7UR7 chain A); fitness is the TM-score of the ESMFold-predicted structure against it, normalised by reference length. The native sequence scores 0.9137 against itself (`results/SEQUENCE_GA_REPORT.md` header).
- **GA design, held fixed across every arm in every experiment cited here.** Population 16, 20 evaluated populations (generation 0 = random start, 19 breeding steps), tournament size 3, elitism 2, crossover rate 0.9, mutation rate 0.05, length bounds [30, 80]. Where an arm injects extra genomes per breeding step (agents, circles, random immigrants, the oracle), the comparison is always read at the smallest distinct-fold count reached by any arm in that seed (`n_cut`), never at each arm's own larger budget.
- **Model.** `gemma4:12b`, temperature 0.7, Ollama, wherever an LLM is used at all.
- **The rule for "beats".** Fixed before any result was read, used identically across every comparison cited here: X beats Y only if X's best-so-far fitness at `n_cut` is higher than Y's in *every* seed compared. With `s` seeds the smallest two-sided sign-test p reachable is `2 / 2^s` (0.0625 at 5 seeds, 0.25 at 3), so no comparison below reaches significance at 0.05; "beats" is read against this fixed, conservative rule, not a p-value.

## Evidence 1. The five-arm sequence comparison: LLM operators, circles and random injection are indistinguishable

Random search (A), GA with deterministic operators (B), GA with LLM operators — `position` mutate, `segment` crossover — (C), GA with LLM operators plus circles (D), and GA with LLM operators plus random immigrants, the control for D (E). 5 seeds each, ~275–358 distinct folds per run depending on arm (`results/SEQUENCE_GA_REPORT.md` §1, §9.0–§9.1).

| comparison | result | source |
|---|---|---|
| C beats A | 5 of 5 seeds, mean +0.0862 | SEQUENCE_GA_REPORT.md §1 |
| C vs B (deterministic) | 3 of 5, mean +0.0234 — no demonstrated difference | SEQUENCE_GA_REPORT.md §1 |
| D (circles) vs C | 2 of 5, mean +0.0037 — no demonstrated difference | SEQUENCE_GA_REPORT.md §9.0 |
| D vs E (random immigrants) | 1 of 5, mean +0.0053 — no demonstrated difference | SEQUENCE_GA_REPORT.md §9.0 |
| E vs C | 2 of 5, mean −0.0016 — no demonstrated difference | SEQUENCE_GA_REPORT.md §9.0 |

GA beats random search, which is the only demonstrated effect in this arm set. Nothing distinguishes deterministic operators from LLM operators, or LLM operators from LLM operators plus a coordination architecture, or that architecture from plain random injection. Circles measurably raise population diversity over C in every seed (mean pairwise edit distance +11.1 to +21.9) without that diversity converting into a fitness edge; the more diversity a seed gained, the worse it tended to do (rank correlation −0.90 over 5 points — a description, not a test) (SEQUENCE_GA_REPORT.md §9.11).

## Evidence 2. The memory arms: behaviour changed exactly as predicted, fitness did not move

Three arms give each of 2 persistent agents a private record of its own past position edits and their fitness outcome: M1 keeps the record but never shows it (the baseline for "what this agent does without seeing its history"), M2 shows the record as prose ("round N: position p old→new; fitness X → Y (better/worse)"), M2b shows the same real record converted to two explicit lists ("Do not change these positions: …" / "These positions improved before: …"). 3 seeds each, 38 agent-propose calls per arm per seed (2 agents × 19 breeding steps) (`results/AGENT_MEMORY_STAGE3_TABLES.md`, `results/AGENT_MEMORY_STAGE3B_TABLES.md`).

**The mechanism was isolated before the GA arms were built**, in two isolated probes (no GA, no fitness search — one agent, paired calls, 100 per condition):
- A gate (`results/raw/agent_memory_gate.json`) showed a synthetic record changes the position distribution at all: total-variation distance 0.382 against a null of 0.306 (two same-size draws from the pooled histogram), paired-permutation p = 0.0016.
- A format probe with a *known* right answer (`results/raw/agent_memory_format.json`) then showed prose and explicit lists pull in different directions. Against a no-record null of 6/100 landing on a to-be-avoided position and 3/100 on a to-be-preferred one: the prose record *raised* the avoided share to 24/100 (exact-sign p = 0.0003) and left the preferred share unmoved at 3/100 (p = 1.0); an explicit avoid list dropped the avoided share to 0/100 (p = 0.031) and an explicit prefer list raised the preferred share to 96/100 (p ≈ 2×10⁻²⁸).

**Inside the real GA runs, behaviour changed as the isolated probes predicted, plus one related effect the probes did not test, with one number needing care to read correctly.** Fallback rose exactly as the format probe predicted, and edits spread over far more of the genome once the explicit-list fix was in place:

| what changed | M1 (baseline) | M2 (prose) | M2b (explicit lists) | source |
|---|---|---|---|---|
| agent-propose fallback, worst seed | 0/38 | **20/38** (seed 1) | 0/38, every seed | STAGE3_TABLES.md, STAGE3B_TABLES.md §1b |
| distinct positions edited per seed (not directly predicted by the isolated probes, but consistent with fewer no-op/fallback calls leaving more genuine edits to spread over) | 17, 15, 15 | 27, 25, 23 | **35, 34, 22** | STAGE3_TABLES.md, STAGE3B_TABLES.md §5 |

For where the edits land, the raw counts (STAGE3B_TABLES.md §6, gen 2+, pooled over 3 seeds) are M1: 187/255 = 73% of edits land on a position in the agent's own record at all, split 92 "better"-labelled / 95 "worse"-labelled; M2: 97/174 = 56% land in-record, split 40 / 57; M2b: 18/255 = 7% land in-record, split 16 / 0. Read as raw shares of *all* edits, M2's worse-rate (57/174 = 33%) is numerically *below* M1's baseline (95/255 = 37%) — because M2 touches record positions at all less often (56% vs 73%), not because it specifically avoids "worse" ones. **Read as a share of the edits that do land in-record — the number the format probe's controlled design actually measured — the effect is there and correctly signed:** M1 splits those touches 49%/51% between better/worse, essentially even; M2 splits them 41%/59%, worse outweighing better; M2b splits its rare in-record touches 89%/0%, almost entirely on "better"-labelled positions and never on a "worse" one. The controlled, paired version of the same test (the Stage 2 gate, synthetic record, one agent, 98 paired calls, `results/raw/agent_memory_gate.json`) shows the same direction without this artefact, because there the model was shown a record on every paired call: landings on "better"-labelled positions fell from a matched null of 30/98 to 8/98 (exact-sign p = 0.0002), and landings on "worse"-labelled positions rose from 30/98 to 47/98 (p = 0.016).

M2's prose record made the model more likely to fall back (a same-letter, no-op answer, rejected by the parser after retries) and, conditional on touching a record position at all, more likely to land on one it had just called bad. M2b's explicit lists fixed both: fallback to 0 in every seed, and — this part unambiguous even as a raw share of all edits — 0 of 255 pooled edits landed on a "worse"-labelled position, against M1's 95 and M2's 57.

**And fitness never moved.** Best TM-score at the common evaluation count, mean over 3 seeds: M1 0.5065, M2b 0.5047, M2 0.4707 (STAGE3B_TABLES.md §2). None of M1, M2 and M2b demonstrably beat each other under the fixed rule: M1 higher than M2 in 1 of 3 seeds (mean +0.0359); M1 higher than M2b in 1, lower in 1, tied in the third (mean +0.0019, essentially zero); M2 higher than M2b in 1 of 3 (mean −0.0340). The intervention that most cleanly did what it was designed to do (M2b: obeyed almost perfectly, confirmed inside the real search) produced no measurable fitness advantage over the intervention that could not see its own history at all.

## Evidence 3. The oracle arm O: direct fitness access, worst of the five

Arm O replaces the LLM-driven agents with 2 slots run by a greedy hill-climber, no LLM at all: at each breeding step, each slot tries 8 single-position mutations of its own current genome, scores *each one* with the real fitness function — through the same cache-aware evaluator the rest of the population uses, so every candidate is charged to the same distinct-fold budget as everything else, not free information — and keeps whichever of {its current genome, the 8 candidates} scores highest. It never accepts a worse move. 3 seeds (`results/AGENT_MEMORY_STAGE3C_TABLES.md`).

| arm | mean best TM at common budget | range | rank |
|---|---|---|---|
| M1 | 0.5065 | 0.4623–0.5355 | 1 |
| M2b | 0.5047 | 0.4573–0.5355 | 2 |
| M3 (random immigrants) | 0.4893 | 0.3970–0.5977 | 3 |
| M2 | 0.4707 | 0.3781–0.5646 | 4 |
| **O (oracle)** | **0.4271** | 0.4120–0.4384 | **5, last** |

O is not merely tied with the field — it is beaten, under the fixed pre-set rule, by two of the four other arms: M1 beats O in 3 of 3 seeds (mean +0.0795), M2b beats O in 3 of 3 seeds (mean +0.0776) (STAGE3C_TABLES.md §2). M3 (random injection) does not formally beat O under the strict rule (higher in 2 of 3 seeds) but leads it on the mean by +0.0622. No arm beats M1, M2b or M3; only O is beaten.

## Evidence 4. The mechanism: single-position substitution is a weak move on this landscape

O's own within-step search record explains why direct fitness access did not help. Per seed, over 36 steps (2 slots × 18 breeding steps with a known base fitness — generation 0 starts from a fresh random genome, so it contributes no step) and 288 candidate evaluations:

| seed | steps | kept its current genome (none of 8 candidates beat it) | moved | mean gain when it did move |
|---|---|---|---|---|
| 0 | 36 | 22 (61%) | 14 (39%) | +0.0206 TM |
| 1 | 36 | 22 (61%) | 14 (39%) | +0.0075 TM |
| 2 | 36 | 19 (53%) | 17 (47%) | +0.0114 TM |

(STAGE3C_TABLES.md §1c, computed from `results/raw/agent_memory_O_seed{0,1,2}.json`.) In 53–61% of steps, *none* of 8 candidate single-position substitutions beat the genome already held — the oracle, with a real fitness readout on every try, found nothing worth taking. When it did find something, the gain was small: 0.0075 to 0.0206 TM-score, against a scale where the best fitness at the common evaluation count, across every arm and every seed in Evidence 1–3, sits between 0.3587 (`results/raw/sequence_ga_cmp_D_seed0.json`) and 0.6372 (`results/raw/sequence_ga_cmp_D_seed1.json`). A move class this weak cannot be rescued by better *choosing* among its own candidates — the ceiling on what one single-position edit can buy, even chosen by an oracle, is low relative to what the search needs.

## Evidence 5. The budget argument: search-within-a-step is not free, and it did not pay for itself

Every one of O's 288 candidate evaluations per run is a real ESMFold fold, charged to the same distinct-fold budget the comparison is read at. That budget is spent once, and O spent a large share of it searching *within* single breeding steps rather than advancing *across* them: at the common evaluation count each seed is cut to, O had completed only about half the other arms' breeding steps.

| seed | n_cut (common budget) | breeding steps O completed by n_cut | breeding steps M1/M2/M2b/M3 completed (all reach n_cut at the end of the run) |
|---|---|---|---|
| 0 | 309 | ~10 of 19 | 19 of 19 |
| 1 | 298 | ~9 of 19 | 19 of 19 |
| 2 | 318 | ~10 of 19 | 19 of 19 |

(Recomputed from each run's `generations[i].distinct_evaluations_so_far` against the seed's `n_cut`, `results/raw/agent_memory_O_seed{0,1,2}.json` against `results/raw/agent_memory_M1_seed{0,1,2}.json`.) O also paid an extra GPU residency swap every breeding step (moving ESMFold back onto the GPU after the LLM-driven fill's crossover/mutate calls, then off again for the next generation's fold phase) that the other arms do not pay, a real but secondary cost next to the evaluation-budget one. At equal distinct-fold budget, the choice was between roughly 19 generations of ordinary selection with a cheap or free per-step move, and roughly 9–10 generations of the same selection with a locally-optimal-by-construction move. The first choice won in every case tested.

## A screen, not an arm: does a protein language model's preference predict fitness here?

Evidence 1–5 are all GA arms. This is not: it is a measurement over moves that were already made, and it is reported here because it bears directly on the obvious objection to everything above — that the operators tried so far were simply not well enough informed about proteins.

A later branch (`results/MOVE_CLASS.md`) ran four deterministic move classes plus the baseline operator with no LLM anywhere, and recorded, for 3,990 moves, the base genome, the child genome and both fitnesses. Every one of those moves was then scored with ESM-2 150M's masked likelihood (`facebook/esm2_t30_150M_UR50D`, 148,140,154 parameters; `results/ESM2_LIKELIHOOD_SCREEN.md`). **No model proposed any of these moves**, so they are an unbiased sample with respect to the model — a stricter test than scoring moves it chose itself. Two quantities: **Δlogit**, the model's preference for the substitution in the base's own masked context, over the 3,192 equal-length moves; and **ΔPLL**, the whole-sequence likelihood change, over all 3,990 including the 798 indels.

| metric | n | pooled Pearson r | moves the model prefers that improved | moves it dislikes that improved | sign agreement |
|---|---|---|---|---|---|
| Δlogit | 3192 | +0.025 | 514/1626 (31.6%) | 480/1533 (31.3%) | 49.6% |
| ΔPLL | 3990 | −0.020 | 630/1981 (31.8%) | 668/1976 (33.8%) | 49.0% |

**The model's preference does not predict fitness on this landscape.** The pooled correlations are about 0.05% of variance; the per-arm signs disagree on ΔPLL (S1, S3 and S4 negative, S2 and B positive); and on ΔPLL the decision-relevant gap runs the wrong way. Sorting moves by this model's preference separates them less than the choice of move class does. A genome's own likelihood does track its fitness weakly (+0.146 pooled over 3,197 bases), but that is absent in one of the three seeds, partly a trajectory artifact, and about genomes rather than about ranking moves. For calibration, all 3,197 evolved genomes score below the native 7UR7 sequence (−3.111 mean against −2.253), so this is measured in a band the model finds uniformly unlikely. (p-values there are an uncorrected Fisher-z normal approximation over 20 tests; sign consistency is what to read.)

**What this does and does not add.** What was measured is that this operator-side signal does not rank already-made moves by their realised fitness gain — consistent with the claim below, but adding no search result of its own, since no search was run here. It is **not** a demonstration that a protein language model cannot be a useful operator here: the screen measures the bulk of the distribution, whereas an argmax operator would propose its extreme selected tail, and only one model size was tried. No ESM-2 arm has been run.

## The conclusion, stated plainly

On this problem, at this budget, **the choice of perturbation operator does not determine search quality; something else — most plausibly selection acting over many generations — is doing the work that matters.** Four interventions (M1, M2, M2b, and circles in the earlier five-arm comparison) changed operator behaviour in exactly the way each was designed and predicted to, and none of them moved the search's outcome. The fifth intervention (O) did not merely fail to help: giving an operator direct, real access to the fitness function and having it act as a true greedy hill-climber made the search worse, because the budget that access consumed was budget the other arms spent on more generations of selection instead — and generations of selection, in this experiment, mattered more than local move quality.

## Scope, stated plainly

This is: **one target** (a single 63-residue reference, PDB 7UR7 chain A), **one fitness function** (TM-score of an ESMFold-predicted structure against that one reference), **one move class** (single- or few-position substitutions to a fixed 20-letter amino-acid alphabet — no insertions, deletions, or larger structural moves were tried by any arm), **population 16, 20 evaluated populations** (19 breeding steps) in every arm cited here, **3 to 5 seeds** depending on the experiment (5 for the arm-A–E comparison, 3 for the memory and oracle arms), and **one main model** (`gemma4:12b`) for every LLM-driven arm. It says nothing about:
- other target proteins, other fitness landscapes, or problems where single-position moves carry more signal than they evidently do here;
- larger populations, more generations, or a different evaluation budget — the oracle's loss is specifically a budget-allocation result (Evidence 5), and a much larger budget, or a budget where within-step search is cheaper relative to a generation, could change which allocation wins;
- larger or different LLM models — one model family was used throughout, at one temperature;
- other move classes — a real greedy oracle over a richer move set (multi-position edits, structural moves, or a smarter local-search budget than 8 single-position tries) was not tested, and Evidence 4 only shows that *this* move class is weak here, not that no move class could do better. Since this document was written, multi-position, segment and indel moves *were* run in a plain GA with no LLM (`results/MOVE_CLASS.md`): they changed how often a move improves on its base, in a consistent direction across seeds, but produced no fitness ordering among the move classes at the same budget — which extends the claim below rather than qualifying it. They were not tried as an oracle's candidate set, so the oracle-specific question stays open;
- whether a better-informed *proposal signal* would help — the screen above found none in ESM-2 150M's likelihood, but that is a measurement over moves the model did not propose, at one model size, so no protein-language-model arm has actually been run;
- whether selection itself, which this document credits with "doing the work," was ever isolated or ablated — no experiment cited here holds crossover and mutation fixed while varying only selection, so that attribution is a reading of the pattern across five arms, not a direct test.

## Limits, stated plainly

- **No operator ablation.** Nothing here isolates selection's own contribution by holding it fixed and varying only the operator in a design built for that purpose; the "selection is doing the work" reading follows from every operator-side intervention failing to move fitness, not from a positive measurement of selection's effect.
- **The oracle's search width (8) and move class (single-position) were fixed, not swept.** Evidence 4 and 5 together support a *budget* explanation for O's loss, but a wider or cheaper search, or a stronger move class, was not tried; the document does not claim 8 single-position candidates is the best an oracle-style upper bound could do here, only what this one did.
- **The ESM-2 screen is a screen, not an arm.** It scores moves produced by other operators, so it bounds what that model's likelihood *ranks*, not what an ESM-2 operator would *do*; an argmax proposal is the extreme selected tail of the distribution it measures. It used one model size (150M — 650M does not fit alongside ESMFold on this card) and every sequence it scored sits below the native sequence's likelihood, where the model is least likely to be calibrated. Its p-values are an uncorrected normal approximation over 20 tests.
- **Every seed count here is small** (3–5), and by the stated rule no comparison in this document reaches p < 0.05; "beats" and "no demonstrated difference" both describe a fixed decision rule applied to a small sample, not a statistical conclusion.
- **The memory arms' mechanism is well-isolated (Evidence 2's two probes plus the in-loop confirmation); the fitness null is not.** Showing that M2b's record is obeyed and M1/M2/M2b are statistically indistinguishable on fitness is not the same as showing memory *cannot* help under any format, window size, or content; only prose vs. two explicit-list variants at one window size (8 entries) were tried.
