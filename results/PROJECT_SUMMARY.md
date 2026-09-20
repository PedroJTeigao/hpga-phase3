# Project summary: HPGA on an LLM substrate, Phase 1 through the five-arm sequence comparison

*Every number carries its source in brackets. **P2** = `results/PHASE2_RESULTS.md`, **P3** = `results/PHASE3_RESULTS.md`, **SEQ** = `results/SEQUENCE_GA_REPORT.md` (its §3 has nine numbered tables, cited as "§3 tbl n"; its §9 is the circles/immigrants section), **ESM** = `esmfold/RESULTS.md`, **RM** = `README.md`, **raw** = a file under `results/raw/`. Where I recomputed a figure from raw files it says so. Writing task only: no experiment was run for this document.*

**Source gap.** `PHASE1_RESULTS.md` is not in this repository or its git history (P2 and P3 refer to it as `Phase 1/results/PHASE1_RESULTS.md`, a sibling folder that is not here). Phase 1 figures below are second-hand, quoted only where a committed file restates them, and marked "via".

## 1. Abstract

The project began with Xue et al.'s bottleneck model for island-model parallel GAs and asked what happens to it when the genetic operators are LLM calls; it widened into whether LLM operators and multi-agent structures improve search at all. LLM operators move the serial fraction from 8.9% to 99.95% [P2 §6], leaving worker count nothing to act on (N=8 speedup 1.078 [P2 §2]); parallel dispatch of operator calls reached 1.00–1.23x against an ideal of 2–8x on this hardware, with the mechanism not isolated [P2 §7.3–7.4]. A prompt format that removes the option of copying (`position`/`segment`) brings fallback to 0 at three alphabet sizes (0/400 and 0/100 at 20 letters [P3 §9.1]), but the operators then use a small part of their space: 20 of 63 mutation positions and 1–2 of 99 cut values [P3 §9.2–9.3]. Agent and circle structures produce reliably higher population diversity (1.45x–5.95x [P3 §2, §6.2]) that did not become a fitness lead on the lattice [P3 §7.3], and in sequence mode circles did no better than random immigrants (D above E in 1 of 5 seeds [SEQ §9.0]). On TM-score fitness, GA with LLM operators beat random search in 5 of 5 seeds (+0.0862 mean) but was not distinguishable from GA with deterministic operators (3 of 5 seeds) [SEQ §1], at 4.0–5.4x the wall time [SEQ §4]. With 5 seeds the smallest reachable p is 0.0625 [SEQ §1], so no sequence-mode comparison is significant, and parallel scaling was never measured in sequence mode.

## 2. The original question

The base paper is Xue et al. (NoCS 2014). Its bottleneck model (Eq. 3/7) gives the speedup ceiling of the fitness-dispatch (DIS) phase, and its DIBM model (Eq. 11) says capacity and speedup scale with P borrowed injection channels [P2 §2, §7]. Phase 1 established the DIS-phase ceiling on a classical GA [P2 §2]. Phase 2 put LLM calls in place of crossover and mutation for one stated reason: "to put `T_calc`/`T_turnaround`-scale latency into a large, realistic regime and measure what that does to the bottleneck model," not to improve search, and it predicted no semantic advantage on a flat string of move symbols [P2 §5].

The question widened in three steps, each documented in the file that ends the previous one. P2 §8 added persistent-identity agents and communication. P3 added circles and a blackboard, and ended by arguing that the diversity-to-fitness question could not be settled on a representation with no semantic content, naming amino-acid sequences scored by ESMFold as the critical path [P3 §8]. SEQ then asks the search question directly: random search vs GA vs GA with LLM operators (A–C, [SEQ §1]), and circles vs random immigrants (D, E, [SEQ §9]).

## 3. What each phase built, measured and established

### 3.1 Phase 1: classical HPGA, Eq. 3/7 (second-hand)

Classical island-model GA on the 2D square-lattice HP model, with a DIS/GA timing harness (`island.py`, `worker.py`, `instrumentation.py`) and a `pop_size=256` sweep [P2 §2, RM]. Figures restated elsewhere: serial fraction 8.9% with deterministic operators [via P2 §6]; Eq. 3/7 describes the DIS-phase ceiling and says nothing about the GA phase [P2 §2]; on Unger & Moult n=20 (E\*=-9) the GA reached the exact optimum in 3/5 seeds and landed 1 contact short in the other 2, converging by generation ~25–50 [via RM, 3D migration section, quoting P1 §2]; wall-clock timing drifts ~6% between sessions [via P2 §7.3.1]. **Established (as restated elsewhere):** the DIS-phase speedup ceiling and its dependence on the serial fraction, and a working 2D baseline. Nothing more about Phase 1 can be sourced from committed files.

### 3.2 Phase 2: LLM operators, serial fraction, latency, affordance (`gemma4:12b`, Q4_K_M)

**Built:** `HPGA_OPERATOR_MODE=llm` swaps crossover and mutation for Ollama calls; the rest of the harness is byte-identical to Phase 1 [P2 §1, RM]. **Measured** (details in findings 1–5, 7, 8): a paired pilot and N-sweep at `pop_size=8`, a 33-call latency fit, a diff-style prompt, operator reliability and coverage, parallel operator dispatch timed on `llama3.2:1b`, and a 3-arm agent run. Key numbers: GA phase 305.26 s of 305.42 s wall (99.95%), N=8 speedup 1.078 [P2 §2]; `latency_s ≈ 1.733 + 0.00537·in + 0.339·out`, R²=0.991 (I refit it from the raw file and got the same constants) [P2 §3]; free-form mutate a no-op in 150/150 calls, and `position`/`segment` formats at 0/50 fallback each but with 80% of mutations on positions 0–1 and 36 of 40 cuts at the midpoint [P2 §4]; dispatch speedup at most 1.23x [P2 §7.3]. Also the 3D model check: plain 3D HP on CPSP-tools S1 (n=27, proven E\*=-22), best GA result 20 over 5 seeds, validation not passed, external structure check unfinished; the base paper uses HPSC, which this model is not [RM]. **Established:** worker count is stranded under LLM operators; free-form operators are unreliable, and position/segment operators are reliable but low-variance.

### 3.3 Phase 3: circles, blackboard, diversity gap (lattice)

**Built:** circles of agents that consult within a circle, and a blackboard that carries observations, never genomes, across circles [P3 §1]. **Measured:** population diversity, identical-parent rate, and fitness, on an anchor run (length 18, 1 seed) and on length 24 with 3 seeds at 10 and 15 generations. Key numbers: diversity ratios 1.45x–3.16x over six runs and 5.95x (12.97 vs 2.18) at length 24 [P3 §2, §6.2]; final fitness 3.33 vs 3.67 at 10 generations and 3.67 vs 4.00 at 15, with seed 1 an outright loss [P3 §6.3, §7.2]; 1.89x more tokens per fitness point [P3 §6.4]; ~4.8x wall time (3533 s vs 731 s) [P3 §6.6]. Two by-products: `CROSSOVER_MIN_DIFF=2` rejected 17.6% of 5000 correct crossovers [P3 §4], and a free deterministic sweep put the plateau at generation ~54 where the real arm plateaued near 7 [P3 §6.5]. **Established:** a robust diversity effect and no demonstrated fitness benefit at roughly 2x token cost; P3 §8 concluded the representation, not the architecture, was the limit.

### 3.4 P3 §9: 20-letter operator coverage and the segment ablation (no fitness, no GA)

**Built/measured:** the sequence `GenomeModel`'s LLM operators called in isolation, `gemma4:12b`, temperature 0.7, seed 0, each figure a single run [P3 §9]. Key numbers: fallback 0/400 (`position`), 0/100 (`segment`), 24/30 and 26/30 (`full` styles) [P3 §9.1]; mutate reached 20 of 63 positions, one alone 37.3%, χ²=4324.37 (df 62) [P3 §9.2]; crossover, with 99 possible cuts, used 2, 2, 2 and 1 distinct values across four 100-call prompt settings (40×79/50×21; example moved to 0-70, 40×74/50×26; no example, 40×77/50×23; prose number 25, no example, 50×100) [P3 §9.3; raw `segment_example_ablation_summary.json`, checked]. **Established:** the format effect and the low-variance behaviour both carry to 20 letters; the 40 came from the prompt's prose number, not the worked example.

### 3.5 Sequence GA: TM-score fitness, arms A–E, five seeds

**Setup.** Fitness is the TM-score of the ESMFold structure against 7UR7 chain A (63 residues), normalised by reference length; the native sequence scores 0.9137 against itself [SEQ header; raw `tm_fitness_sanity_console.log`: 0.91366]. ESMFold runs fp32 on a 15360 MiB T4 (fp16 gives 100% NaN; fp32 OOMs at 300 residues) [ESM §3–4]. Population 16, 20 generations, ~275–282 distinct folds for A–C; D and E ran 342–358 because their 4 injected genomes per step add folds, so every comparison is read at the common per-seed count (275–280) [SEQ §2, §9.1]. Arms: A random search; B GA, deterministic operators; C GA, LLM operators (`position` mutate, `segment` crossover); D = C plus circles ported to sequences; E = C plus 4 random immigrants per step, the control for D [SEQ §2, §9].

| arm | mean best TM at common cut (5 seeds) | range over seeds | source |
|---|---|---|---|
| A random | 0.4002 | 0.3859–0.4104 | [SEQ §9.2] |
| B GA | 0.4630 | 0.4007–0.5040 | [SEQ §9.2] |
| C GA+LLM | 0.4864 | 0.4370–0.5434 | [SEQ §9.2] |
| D GA+LLM+circles | 0.4901 | 0.3587–0.6372 | [SEQ §9.2] |
| E GA+LLM+immigrants | 0.4848 | 0.4164–0.6099 | [SEQ §9.2] |

Rule fixed in advance: X beats Y only if higher in every seed. B vs A: B higher in 4 of 5 (seed 3: A by 0.0028), no demonstrated difference. **C beats A: 5 of 5, mean +0.0862.** C vs B: 3 of 5, mean +0.0234, no demonstrated difference [SEQ §1]. D vs C: 2 of 5, mean +0.0037; D vs E: 1 of 5, mean +0.0053; E vs C: 2 of 5, mean -0.0016 [SEQ §9.0]. I recomputed every per-seed final and common-cut value from `sequence_ga_cmp_{A..E}_seed*.json`; all matched. **Established:** GA beats random search in most seeds; LLM operators, and circles on top of them, are not distinguishable from their controls with this design. Every final genome is 52–67 edits from the reference [SEQ §9.9].

## 4. Findings

| # | Claim | Evidence | Strength |
|---|---|---|---|
| 1 | Under LLM operators the GA phase is ~all wall time; `n_workers` is stranded | 305.26/305.42 s; N=8 speedup 1.078 vs 1.001 predicted [P2 §2]; serial fraction 8.9% → 99.95% [P2 §6] | One sweep, `pop_size=8`, 3 generations; follows from the architecture |
| 2 | Output tokens cost ~63x input tokens on `gemma4:12b`/CPU, and that was CPU decode | 0.339 vs 0.00537 s/token [P2 §3]; 0.01417 s/token on 1b/GPU [P2 §7.1] | 33 calls per fit, two models, temperature 0 |
| 3 | Diff-style prompt helps mutate, hurts crossover | -26% vs +27% op time; crossover valid 0.62 [P2 §3] | Single small paired run (9–18 requests) |
| 4 | Free-form formats are copy-prone; `position`/`segment` remove fallback | 150/150 no-op [P2 §4.1]; 0/50 [P2 §4.3]; 0/20 [P2 §7.6.1]; 0/400, 0/100 [P3 §9.1] | Three alphabets, one model; 0/100 bounds the rate at ~3%, 0/400 at ~0.75% [P3 §9]; fails on `llama3.2:1b` [P2 §7.1] |
| 5 | Reliable operators use a small part of their space | 80% on positions 0–1 [P2 §4.4]; 20 of 63 positions [P3 §9.2]; 1–2 of 99 cuts [P3 §9.3]; cuts 40/50 only in every C run [SEQ §4] | Three alphabets plus 5 in-loop runs; one model |
| 6 | The cut mode 40 comes from the prompt's prose number; with none, the midpoint every time | 79/74/77% at 40, then 50 in 100/100 [P3 §9.3] | Four single 100-call runs; shifted example plus prose 25 not run |
| 7 | Parallel operator dispatch scaled 1.00–1.23x against an ideal of P, and a larger `OLLAMA_NUM_PARALLEL` slowed even P=1 | pop 8: 1.00/1.15/1.18/1.23; pop 32: 1.00/1.00/1.04/1.08 [P2 §7.3]; 79.96 s vs 35.52 s [P2 §7.2] | 3 repeats, adjacent points within ~1 sd; 1b as instrument; mechanism not isolated (§5) |
| 8 | Agent communication bought nothing measurable; refine is an echo fixed point | same trajectory for +0.5% tokens [P2 §8.1]; identical output 6/6 generations [P2 §8.3] | One 3-arm run, `pop_size=5`, 6 generations |
| 9 | Agent/circle structures raise diversity and it never inverted | 1.45x–3.16x, six runs [P3 §2]; 5.95x [P3 §6.2]; 15-gen final 11.35 vs 2.29 = 4.97x (recomputed from raw) | 7 stated ratios (P3 §8 says eight); 3 seeds at length 24. In sequence mode D > C in 5/5 seeds but D ≈ E [SEQ §9.11] |
| 10 | On the lattice, diversity did not become a fitness lead | 3.33 vs 3.67 (10 gen), 3.67 vs 4.00 (15 gen, one seed lost) [P3 §6.3, §7.2]; ~1.9–2.1x worse per token [P3 §5, §6.4] | 3 seeds; P3 counts 2 wins, 1 loss, 3 ties [P3 §7.3] |
| 11 | GA with LLM operators beat random search in every seed, not deterministic GA | C>A 5/5 (+0.0862); C vs B 3/5 (+0.0234) [SEQ §1] | 5 seeds, min p 0.0625, not significant; no operator ablation [SEQ §5] |
| 12 | Circles did not beat random injection | D vs E 1/5; D 0.4901, E 0.4848, C 0.4864 [SEQ §9.0]; injected 0.261 vs 0.250 [SEQ §9.11] | 5 seeds; D's spread (0.3587–0.6372) far exceeds the gaps |
| 13 | No sequence search approaches the reference | 52–67 edits in every arm; TM >0.5 in 2/5 B, C, D, 1/5 E, 0/5 A [SEQ §9.0, §9.9] | 5 seeds each |
| 14 | LLM arms cost 4–5x wall time at equal distinct folds | C/B 4.0–5.4x; D/C 1.50x [SEQ §4, §9.11] | 5 seeds |
| 15 | fp16 ESMFold is unusable; fp32 wall is 250–300 residues | 100% NaN at 50–250; OOM at 300 [ESM §3–4] | One measurement per length; deterministic |
| 16 | The min-diff crossover gate measured the wrong thing | 17.6% of 5000 correct crossovers rejected [P3 §4] | 5000 samples, no model |

## 5. Negative results, stated plainly

**Did not work, or no demonstrated benefit.**
- LLM operators vs deterministic operators in sequence mode: C higher in 3 of 5 seeds; without seed 3, C's mean difference from B is -0.0064 [SEQ §4]. On the lattice, the LLM crossover reproduced the midpoint single-point cut the deterministic operator already makes [P2 §5].
- Circles: no fitness edge over C or over random immigrants [SEQ §9.0], and on the lattice no widening lead with more generations [P3 §7.2]. Adding seed 4 shrank D's mean edge over C from +0.0135 to +0.0037 and over E from +0.0184 to +0.0053 [SEQ §9.13]. D holds both the best single run (0.6372, seed 1) and the worst of the LLM arms (0.3587, seed 0) [SEQ §9.0].
- Communication paid nothing and refine added zero distinct genomes [P2 §8]. Diff-style crossover made things worse [P2 §3]. The affordance fix failed on `llama3.2:1b` [P2 §7.1].
- The 3D model's validation did not pass (20 of 22) and its external check is unfinished [RM].
- Proxies mispredicted: the deterministic sweep put the plateau at generation ~54, reality ~7 [P3 §6.5]; ESM's lattice proxy predicted a 32.8% revisit rate [ESM §5] and the sequence runs had 11.9–14.1% cache hits (recomputed from `n_cache_hits`, B and C; a different config, so not like-for-like).

**Claims withdrawn or shrunk.**
- **Identical-parent feedback loop.** The off/on identical-parent ratio was 2.7x while an operator was broken and 1.2x once both worked (0.200 vs 0.167; 1.51x call-weighted); a middle 1.65x figure has unknown provenance and is excluded. "A hypothesis that looked strong and shrank" [P3 §3]. I recomputed 0.200 and 0.1667 from `circles_smoke_summary_1789603059.json`.
- **Example anchoring.** The 40-mode was not produced by the worked example: with the example moved to 70, 40 stayed modal (74%) and 70 was never chosen; with no example, 77%. This does not show the example could never matter [P3 §9.3].
- **Contention theory.** Two parts. (i) The single-run reading that concurrent dispatch was actively harmful (0.80x at P=4, 0.81x at P=8) did not reproduce with 3 repeats; that P=1 draw was ~2.4 sd above the repeated mean [P2 §7.3.1]. (ii) The account that the flat curve is contention for the shared GPU is still stated in P2 §7.3.2 and §7.4 ("confirmed by direct measurement"), but what was measured is that a larger population also stayed flat. §7 reports no direct measurement of contention, and its concurrency probe, on tiny 16-token requests, reached 6.98x at K=8 on the same GPU and model [P2 §7.1–7.2]. *No committed file records a withdrawal of (ii); if it has been withdrawn, PHASE2_RESULTS.md should say so. Here it is treated as unproven.*
- "Circles reached fitness 2.0 faster" is not carried as an advantage: both arms plateau at the same value [P3 §6.3].

**Where sources disagree.**
- The same comparison (aggregate final fitness at 10 generations, 3.33 vs 3.67) is "deliberately not written as a win" in P3 §6.3 and counted as a win in P3 §7.3 ("two wins, one outright loss, three ties"). P3 §5's table calls generation-reached "marginal" (gen 4 vs 3); §7.3 counts it a tie.
- RM says "Not a git repository"; P3 §5 records git history, a backup and a GitHub remote. RM is stale.
- P3 §8 cites eight diversity ratios; seven are stated (§2, §6.2).

## 6. Limitations and gaps

- **Parallel scaling was not measured in sequence mode.** The sequence drivers run in-process with one worker, outside `Island`: ESMFold is ~13.7 GB in fp32 and fits on the GPU once, so it cannot be replicated per worker [`experiments/run_sequence_ga_smoke.py` docstring; `experiments/run_sequence_ga_comparison.py` docstring; SEQ §6 item 8]. Ollama (~8 GB) and ESMFold also cannot share the 15 GB T4, so arms C, D, E swap them, costing 7–15% of wall time [SEQ §3 tbl 6, §9.7]. The only scaling data are lattice ones [P2 §2, §7.3], on different hardware (a 4 GB GTX 1650 Ti in P2 §7, a T4 for the sequence work [ESM]).
- **Small populations.** `pop_size=8` (lattice), 16 (sequence), 20 generations, ~275–282 distinct folds per run [SEQ §2].
- **One model.** `gemma4:12b` at temperature 0.7 [SEQ §5]; `llama3.2:1b` only as a timing instrument [P2 §7.0]. The latency fit is at temperature 0, the operators run at 0.7 [P2 §5].
- **Seeds.** 5 at most; 3 on the lattice, 1 for the anchor, single runs in P2 §4 and §8. With 5, the smallest two-sided sign-test p is 0.0625, so nothing in SEQ is significant at 0.05 [SEQ §1].
- **One target.** A single 63-residue reference; the native sequence scores 0.9137, which is the ceiling [SEQ §5].
- **No attribution.** No ablation of selection, crossover or mutation, so B's and C's edge over A cannot be assigned to an operator [SEQ §5].
- **Budget definition.** Equal distinct folds hides C's ~5x wall time [SEQ §5]; C's swap accounting differs from D/E's (its Ollama reload, est. 139–175 s per run, sits in LLM time) [SEQ §9.7].
- **Model mismatch.** Plain 3D HP, not the base paper's HPSC [RM]; P2 §7 numbers come from the 2D 3-symbol model [P2 §7].
- **Environment.** Shared node; an Ollama fault stopped E seed 4, which was re-run after a restart with no verdict changed [SEQ §9.12–9.13].

## 7. Open questions and possible directions (options, not recommendations)

1. **Measure worker scaling on the sequence task.** (a) One ESMFold copy per worker needs ~13.7 GB each [SEQ §6]. (b) Batching folds inside one predictor fits one GPU but changes what N means relative to Eq. 3/7. (c) Getting fp16 or autocast to work would shrink the predictor, but fp16 currently gives 100% NaN [ESM §4], autocast is untried [ESM §6], and fitness accuracy is at risk.
2. **More seeds on A–E.** Six seeds allow p=0.03125 only if every seed agrees (2×0.5⁶, arithmetic). One C run took 2875–3859 s and one D run 4657–4922 s [SEQ §3 tbl 6, §9.7].
3. **Attribute the GA's edge over random search** by ablating selection, crossover and mutation [SEQ §5]; cheap for deterministic arms, costly for LLM arms.
4. **Fix operator variation**: ban or randomise the modal cut, change temperature, or draw the cut in code and leave the LLM content decisions. Each moves the operator toward the deterministic one, the comparison that has so far shown no difference; P2 §5 notes rewording could shift the numbers.
5. **Give the LLM something to reason about**, such as structural feedback in the prompt. That tests a different hypothesis and risks leaking the target, which current prompts do not name [SEQ §9.14 item 4].
6. **Compare at equal wall time**, which would favour A and B given the 4–5x cost [SEQ §5]; **vary model or population size**, both single points here, at higher cost per call.
7. **Apply the affordance fix to circle proposals** (`propose` fell back 9.2% [SEQ §9.11]); P3 §8 argues it would not change the fitness picture.

## 8. Reproducibility

- **Repo:** `git@github.com:PedroJTeigao/hpga-phase3.git`. `main` and `circles-sequence` are both at `0f3c028`; this file is on `project-summary`, branched from `main`. **Interpreter:** `/scratch/pcanaste/venv/bin/python` (bare `python3` is 3.6). Circles runs pin `PYTHONHASHSEED=0` [SEQ §9.14].
- **Result → commit** (from `git log`): P2 and README `9765713` (P2 edited at `2bafe19`, which also split out P3); length-24 run `f7e4ba7`; 15-generation run `4bbaad6`; ESMFold results `46521b9`, `d4aa2a4`; P3 §9 probe `1b500ff`, logs `78ee70e`, ablation `acc1d22`, `21b9e30`, doc `50f6457` (later edited at `c089c3a`, branch `llm-operator-genome-dispatch`); sequence A–C `8a77a7c` (branch `tm-score-fitness`); circles port `5ecef19`, D/E driver `b9a7d81`, summariser `1b1b9c2`, `6d1a5f7`; D/E results `2f35fdd`; E seed 4 re-run and five-seed update `0f3c028` (branch `circles-sequence`).
- **Key scripts** (`experiments/`): `run_phase2_n_sweep.py`, `probe_latency_vs_tokens.py`, `run_ga_dispatch_sweep.py`, `run_agents_3arm.py`, `run_circles_smoke.py`, `run_circles_diversity_fitness.py`, `probe_sequence_operator_compliance.py`, `summarize_segment_example_ablation.py`, `run_benchmark_validation_3d.py`, `run_sequence_ga_comparison.py` (A–C), `run_sequence_ga_circles.py` (D, E), `summarize_sequence_ga_{comparison,circles}.py`; `esmfold/tm_fitness.py`.
- **Raw data:** `results/raw/sequence_ga_cmp_{A..E}_seed{0..4}.json`, `llm_operator_calls_*.jsonl`, `sequence_ga_{,circles_}report_tables.md`, `segment_example_ablation_summary.json`, `circles_*_summary_*.json`, `latency_vs_tokens.jsonl`.
- **Checked against raw for this document:** per-seed final and common-cut TM for all five arms; the latency fit; ablation cut counts; circles/diversity summary means and wall times; the anchor's identical-parent rates. All agreed with the prose.
