# Scoring the pre-registered predictions for arm C on `qwen2.5:7b`

*This scores `results/PREREGISTERED_QWEN.md`, committed as `10c1a7f` on 2026-09-30, before any qwen run existed. That file is not edited; this file is where the outcome is recorded. Scored 2026-10-01 from the five runs in `results/raw/armC_qwen2.5_7b/` (launched 2026-09-30 18:31, finished 22:05, every seed on attempt 1, no failed seeds).*

## Method

- Every measured value was computed from the per-attempt call logs, in the same way for both arms.
- Gemma's values computed this way reproduce `SEQUENCE_GA_REPORT.md` exactly: cut counts, distinct mutate positions 28 / 26 / 41 / 28 / 33, and invalid attempts 9 / 3 / 28 / 18 / 15.
- **Accepted** calls are LLM calls whose response parsed, so fallbacks are excluded.
- **Positions** are the indices named in accepted mutate responses.
- **Invalid attempts** are failed attempts, retries included, divided by all attempts.
- **Best fitness** is read at `n_cut`, the smaller distinct-fold count of the two runs, per seed, as the prediction file specified.

**Provenance:** all four checks in `experiments/summarize_arm_c_model.py` passed.
- One build served all five runs, the same at the start and end of each: digest `845dbda0ea48ed749caafd9e6037047aa19acfcfd82e704d7ca97d631a0b697e`, Ollama 0.34.0, Q4_K_M, 7.6B parameters.
- The gemma files predate provenance recording.

## Scores

| # | prediction | measured, qwen (seeds 0 / 1 / 2 / 3 / 4) | gemma arm C, same seeds | score |
|---|---|---|---|---|
| **P1** | modal cut 40 in every seed; at least 95% of accepted crossovers contain a cut at 40 in every seed and pooled; above gemma's share in 5 of 5 seeds; at most 2 distinct cut values | modal cut 40 in all five seeds; **100%** of accepted crossovers at 40 in every seed (127/127, 123/123, 118/118, 119/119, 119/119); 1 distinct cut value per seed | 86% / 95% / 93% / 85% / 89% (the rest at 50), 2 distinct cut values | **TRUE** (every part) |
| **P2** | fewer distinct mutate positions than gemma in 5 of 5 seeds; at most 20 per seed (point guess); modal position 0 or 10; share of changes at 0–3 or 10 above gemma in every seed | distinct positions **35 / 45 / 43 / 42 / 46**: *more* than gemma in every seed; modal position 10 in all five; share at 0–3 or 10: 0.463 / 0.228 / 0.382 / 0.391 / 0.339 | distinct 28 / 26 / 41 / 28 / 33; modal 10 / 3 / 10 / 40 / 10; share 0.343 / 0.332 / 0.326 / 0.332 / 0.356 | **FALSE**. The main prediction is reversed in 5 of 5 seeds. The point guess fails in every seed. Only "modal 0 or 10" holds. The share is higher in 4 of 5 seeds, not every seed (seed 1 is lower). |
| **P3** | mutate fallback at most 2 per seed; crossover fallback 0 | mutate fallback **26 / 7 / 8 / 27 / 5**; crossover fallback 0 | mutate 0 / 0 / 0 / 1 / 0; crossover 0 | **FALSE**: the mutate bound fails in every seed. The crossover part holds. |
| **P4** | pooled mutate invalid-attempt rate above gemma's 73/1402 (5.2%); lower confidence: mostly wrong line count or repeated position, not the same-letter failure; crossover invalid attempts 0 | pooled **614/1871 (32.8%)**; per seed 190/430, 107/366, 92/350, 173/412, 52/313; **614 of 614 named the letter already present**; crossover invalid 0 | 73/1402 (5.2%), all same-letter; crossover 0 | Rate: **TRUE** (6.3x gemma's). Mechanism: **FALSE**. Crossover: **TRUE** |
| **P5** | no ordering of qwen against gemma in all 5 seeds, at the common cut | gemma higher in **5 of 5** seeds (table below) | | **FALSE**: an ordering held in every seed, in gemma's favour |

Best TM-score at `n_cut` (from `results/raw/armC_qwen2.5_7b/armC_model_tables.md`):

| seed | n_cut | qwen best@n_cut | gemma best@n_cut | difference |
|---|---|---|---|---|
| 0 | 281 | 0.3778 | 0.4632 | −0.0854 |
| 1 | 280 | 0.4407 | 0.4745 | −0.0339 |
| 2 | 280 | 0.4636 | 0.5138 | −0.0502 |
| 3 | 279 | 0.5038 | 0.5434 | −0.0396 |
| 4 | 265 | 0.4150 | 0.4370 | −0.0220 |

Mean difference −0.0462. Under the fixed rule, **gemma4:12b beats qwen2.5:7b** in this arm. With 5 seeds the smallest reachable two-sided sign-test p is 0.0625, so this is not significant at 0.05. It is the first operator-level ordering in the project to hold in every seed of a 5-seed comparison.

## What the misses mean

**P3 and P4 are one failure, and it was not the one predicted.** qwen does not miscount lines in the GA. It names the letter already at the position, the rule the prompt states explicitly. It did so in 32.8% of attempts, against 5.2% for gemma, and in 73 calls all three attempts failed, so the call fell back to the deterministic per-site mutation.
- The fallback rate was 73 of 1,330 mutate calls (5.5%); in two seeds it was about 10% (26 and 27 of 266).
- So arm C on qwen is not purely an LLM-operator arm: in those seeds about one mutation in ten was the deterministic operator.
- The isolated probes did not show this. qwen gave 0/100 fallbacks and at most 1 same-letter failure per 100 calls at lengths 40–80, with k = 1. In the GA, k = 2–4 and the genomes are partly converged. Why the rate is so much higher there was not tested.

**P2 reversed.** qwen used *more* distinct positions than gemma in every seed, although at k = 1 in the probes it used far fewer (6 vs 17 at length 63). The probe's k = 1 concentration did not carry over to k = 2–4, which the prediction file flagged as its least supported prediction. qwen's mode is still position 10 in every seed.

**P1 held exactly.** qwen's crossover cut 40 in every one of 606 accepted crossovers, as in the probe's 100 of 100.

**P5:** gemma is higher in all five seeds. The design cannot say why. The models differ in more than one thing measured here: qwen's cut has no variation at all, its mutate calls fall back more often, and it spreads edits over more positions. One untested reading is that the extra fallbacks and the zero-variance cut are part of it.

## Seed 4: 265 distinct folds and 55 cache hits

- **Gemma at the cut.** Gemma's best-so-far at 265 folds is 0.4370; it first reached that value at fold 223, so the cut does not change gemma's number. qwen's best at 265 is 0.4150, its final. **The seed still favours gemma, by 0.0220.**
- **Where the 55 hits come from.**
  - **38 are structural:** two elites carried into each of 19 later generations. Every gemma seed has 38–40 hits for this reason.
  - **The other 17** are 9 genomes recreated from an earlier generation and 8 duplicates within a generation.
  - **15 of the 17** trace to qwen giving the **identical answer to an identical base**: the same post-crossover sequence and the same response, reproducing the same child. That happened 15 times in seed 4, against 0–2 in the other qwen seeds and 0–2 in every gemma seed.
  - The number of repeated bases was the same in both arms in seed 4 (58 each). The difference is that qwen answered a repeated base the same way far more often in this seed.
  - The remaining 2 hits were not traced.
- **Effect.** Those repeats cost no folds, so this run used up its 19 breeding steps at 265 distinct folds instead of about 280.

## Scope

- Five seeds; one target; one run per (model, seed).
- Gemma's arm C ran 2026-09-18/19, and its build is inferred, not recorded (manifest dated 2026-09-15, binary dated 2026-09-09).
- qwen ran 2026-09-30 with recorded provenance.
- Wall time is not compared. Ollama loads during the qwen runs ranged 3.98–50.19 s (`results/raw/armC_qwen2.5_7b/RUN_NOTES.md`).
- None of P1–P5 was revised after the data were seen. P2's point guess and P4's mechanism were marked lower-confidence in advance, and both failed.
