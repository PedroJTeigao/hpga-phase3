# Pre-registered predictions: arm C with `qwen2.5:7b`

**This file is a prediction, not a result.** It was written and committed on 2026-09-30, before any arm C run with `qwen2.5:7b` exists. When this was committed, the run's output directory (`results/raw/armC_qwen2.5_7b/`) did not exist, and no qwen call log from a GA run existed. The predictions below will not be edited after the runs. Any correction or change of reading goes in a separate results file that cites this one.

## What will be run

Arm C of `experiments/run_sequence_ga_comparison.py`, unchanged, with `HPGA_LLM_MODEL=qwen2.5:7b`, launched through the provenance wrapper `experiments/run_arm_c_model.py`.

- Seeds 0–4.
- Population 16, 20 evaluated populations (19 breeding steps).
- Crossover rate 0.9, mutation rate 0.05, tournament 3, elitism 2, length bounds [30, 80].
- Mutate style `position`, crossover style `segment`, temperature 0.7, default retries (2).
- The same generation-0 population per seed as gemma's arm C.
- No circles, no blackboard, no agents.

Only the model changes. The comparison is with arm C on `gemma4:12b` (`results/raw/sequence_ga_cmp_C_seed{0..4}.json`, 2026-09-18/19).

## Where the predictions come from, and why they are uncertain

The predictions extrapolate from isolated probes (no GA) in `MODEL_HETEROGENEITY_STEP1.md` and `MODEL_HETEROGENEITY_STEP4.md`. Those probes differ from the GA in three ways.

1. **The probes change one position per call (k = 1). The GA changes k = round(0.05 × length) = 2–4.** No qwen probe ever asked for more than one position.
2. **The probes use fresh uniformly random genomes of fixed length (40, 63 or 80).** In the GA, lengths vary from 30 to 80 and the population converges.
3. **The probes run 100 calls per operator.** The GA runs about 266 mutate and about 120 accepted crossover calls per seed.

How well the probes predicted gemma's own GA behaviour is the best available guide to how far to trust them for qwen:

| quantity | gemma, probe (step 1, length 63, k = 1) | gemma, actual in arm C (seeds 0–4) | did the probe predict it? |
|---|---|---|---|
| modal crossover cut | 40 (82% of cuts), rest at 50 | 40 in every seed (85–95% of accepted crossovers), rest at 50 | yes, closely |
| distinct cut values | 2 | 2 in every seed | yes |
| distinct mutate positions | 17 in 100 calls | 28 / 26 / 41 / 28 / 33 over about 266 calls | direction only: more positions over more calls and larger k |
| mutate fallback | 0/100 | 0, 0, 0, 1, 0 of 266 per seed | yes |
| mutate invalid attempts | 0/100 (1.00 requests per call) | 9/275, 3/269, 28/294, 18/283, 15/281 (1.1–9.5%; pooled 73/1402 = 5.2%) | **no**: the probe under-predicted |
| crossover invalid attempts | 0/100 | 0 in every seed | yes |

Sources for the arm C row: `SEQUENCE_GA_REPORT.md` §3 tables 7 and 9, and the per-attempt call logs (`FITNESS_PROMPT.md` §2 gives the same invalid counts for seeds 0–4).

## Predictions for qwen inside the GA

Each prediction is stated so that the runs can falsify it. "In every seed" is the project's fixed rule. With 5 seeds the smallest two-sided sign-test p is 0.0625, so a 5-of-5 outcome is direction-finding, not significant.

### P1. Crossover cut

- **Probe basis:** qwen cut at 40 in 100 of 100 calls, with 1 distinct cut value (step 1).
- **Prediction:**
  - The modal cut is **40 in every seed**.
  - The share of accepted crossovers containing a cut at 40 is **at least 95% in every seed**, and pooled over seeds.
  - That share is **higher than gemma's arm C share in the same seed, in 5 of 5 seeds.** Gemma's shares were 107/124, 118/124, 109/117, 104/123, 108/122.
  - At most 2 distinct cut values per seed.
- **What would falsify it:** any seed below 95% at 40, or qwen not above gemma in every seed.

### P2. Distinct mutate positions

- **Probe basis:**
  - At k = 1, qwen used 6 positions at length 63 (gemma: 17).
  - It used 4, 6 and 5 positions at lengths 40, 63 and 80.
  - Positions 0 and 10 took 74–95% of its choices.
  - At length 40 it also tended to list positions 0, 1, 2, … in order (see P4).
- **Prediction:**
  - **Fewer distinct positions changed per run than gemma in the same seed, in 5 of 5 seeds.** Gemma: 28 / 26 / 41 / 28 / 33.
  - Point guess, lower confidence: **at most 20 distinct positions in every seed.** No probe measured k > 1, so this number is the least supported prediction in this file.
  - The modal changed position in each seed is **0 or 10**.
  - The share of changed positions at index 0–3 or at 10 is **higher than gemma's in every seed.**
- **What would falsify it:** qwen at or above gemma's distinct count in any seed; for the point guess, any seed above 20.

### P3. Fallback

- **Probe basis:** 0/100 fallbacks for both operators at length 63 (step 1), and 0/100 at lengths 40, 63 and 80 (step 4).
- **Prediction:**
  - Mutate fallback is **at most 2 per seed** (of about 266 calls).
  - Crossover fallback is **0 in every seed.**
- **What would falsify it:** any seed with 3 or more mutate fallbacks, or any crossover fallback.

### P4. Invalid attempts

- **Probe basis:**
  - qwen mutate needed 1.00 requests per call at length 63 in step 1, and 1.25, 1.03 and 1.01 at lengths 40, 63 and 80 in step 4.
  - That is 25/125, 3/103 and 1/101 invalid attempts.
  - At length 40, 24 of its 25 invalid attempts listed positions 0, 1, 2, … in order and gave the wrong number of lines. Only 1 named the letter already present.
  - gemma's GA invalid attempts were all the same-letter kind (73 of 73).
  - Crossover needed 1.00 requests per call.
- **Prediction:**
  - Mutate invalid-attempt rate, pooled over seeds 0–4, is **above gemma's 73/1402 (5.2%).** The GA's lengths reach down to 30, and qwen was worst at the shortest length probed. The probe's under-prediction for gemma points the same way.
  - Lower confidence: most of qwen's invalid attempts will be **wrong line count or repeated position**, not the same-letter failure that made up all of gemma's.
  - Crossover invalid attempts are **0 in every seed.**
- **Caveat stated in advance:** with k = 2–4, listing "0, 1, 2" in order can be a *valid* answer whenever the count matches k. So the enumeration habit may show up as extra use of positions 0–3 (P2) rather than as invalid attempts. Both readings are recorded here in advance, so neither can be chosen after the fact.
- **What would falsify it:** a pooled rate at or below 5.2%.

### P5. Best fitness: not predicted by the probes

The probes say nothing about search outcome. The only prior is the project's earlier results: `OPERATOR_DOES_NOT_MATTER.md` and `FITNESS_PROMPT.md` found no demonstrated difference from any operator change.

- **Prediction:** at the common cut, no ordering of qwen-C against gemma-C holds in all 5 seeds.
- The cut is, per seed, the smaller distinct-fold count of the two runs.
- Stated so it can be falsified: qwen higher in 5 of 5 seeds, or lower in 5 of 5.

## Measured before this file was written (2026-09-30)

- `qwen2.5:7b` manifest digest `845dbda0ea48ed74…`, pulled 2026-09-20, Q4_K_M, 7.6B parameters.
- Ollama 0.34.0; binary last modified 2026-09-09.
- VRAM at `num_ctx` 4096: 4.42 GiB, all on the GPU (Ollama `/api/ps`); `nvidia-smi` delta 4,649 MiB.
- Model load time 181.4 s, against 14.7 s on 2026-09-21. By arm F's rule, the runs will not be launched until three consecutive reloads measure 7–15 s.
