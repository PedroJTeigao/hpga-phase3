# Does the mutate operator read the letter it is changing? A chance-baseline measure from existing logs

*Written 2026-10-01. No new calls. Computed by `experiments/analyse_blind_choice.py` from call logs already in `results/raw/`. The full generated tables, including the all-attempts version, are in `results/raw/blind_choice_tables.md` and `results/raw/blind_choice.json`. This measure was first computed while designing the OLD-slot gate (`results/PREREGISTERED_OLDFIELD.md`); this file is its full write-up.*

## The question

- In the `mutate/position` format, each response line names a position and a NEW letter.
- The prompt forbids naming the letter already there ("the new letter MUST differ…").
- A model that never looked at the current letter would still name it sometimes, by chance, at a rate set by its own letter preferences.
- Comparing the observed rate of these repeats with that chance rate gives a per-data-set measure of whether the model reads the sequence at the position it chooses. This is the content question behind `OPERATOR_BEHAVIOUR.md`, where Evidence 4 found the chosen *position* does not depend on the letter there.

## Method

For every `mutate/position` attempt in a data set, each parsed line (position, NEW) is checked against the sequence in that attempt's prompt.

- **observed:** the share of lines whose NEW equals the letter already at that position.
- **chance, marginal:** for each line, the probability that a letter drawn from this data set's own NEW-letter distribution equals the actual letter there, averaged over lines. It keeps the model's letter preferences and ignores which positions it picks.
- **chance, shuffle:** each attempt's (position, NEW) lines are re-paired with another attempt's sequence from the same data set, over 200 random pairings; positions beyond the borrowed sequence's length are dropped. This keeps letter preferences, position preferences and how the two go together, and breaks only the link to the actual sequence. It is the stricter baseline, and the one used for the readings below. The 2.5–97.5% range of the 200 pairings is reported.
- **ratio** = observed ÷ chance.
  - About 1: the choice behaves as if the current letter were not read.
  - Below 1: the model avoids the current letter, so it must have read it.
  - Above 1: it is drawn toward the current letter.
- **First attempts only** are the primary measure. A retry follows a hint saying the last answer repeated a letter, so retries are not independent. The all-attempts numbers are in the generated tables and agree in direction everywhere.

**Data sets:**
- **GA:** the operator's mutate calls in gemma4:12b arm C (5 seeds), gemma4:12b arm F (3 seeds, fitness line in the prompt) and qwen2.5:7b arm C (5 seeds). Each changes 2–4 positions per call, on the GA's partly converged sequences.
- **Isolated probes:** `MODEL_HETEROGENEITY_STEP1` (four models, length 63) and the `STEP4` length sweep (gemma and qwen, lengths 40 and 80). Each changes 1 position per call, on fresh random sequences.

## Results (first attempts)

| data set | lines | observed | chance (shuffle) [95% range] | ratio vs shuffle | ratio vs marginal |
|---|---|---|---|---|---|
| gemma4:12b arm C, seeds 0–4 pooled | 4,311 | 0.017 | 0.083 [0.075, 0.091] | **0.21** | 0.23 |
| per seed | | 0.014 / 0.004 / 0.029 / 0.021 / 0.015 | | 0.15 / 0.04 / 0.22 / 0.20 / 0.16 | |
| gemma4:12b arm F, seeds 0–2 pooled | 2,387 | 0.047 | 0.094 [0.083, 0.106] | **0.50** | 0.56 |
| per seed | | 0.055 / 0.032 / 0.053 | | 0.43 / 0.31 / 0.51 | |
| qwen2.5:7b arm C, seeds 0–4 pooled | 4,228 | 0.108 | 0.101 [0.093, 0.109] | **1.07** | 1.41 |
| per seed | | 0.174 / 0.089 / 0.082 / 0.153 / 0.050 | | 0.91 / 0.81 / 0.72 / 1.01 / 0.42 | |
| gemma4:12b probe (k = 1, random) | 300 | 0.003 | 0.049 [0.030, 0.073] | **0.07** | 0.07 |
| qwen2.5:7b probe (k = 1, random) | 326 | 0.006 | 0.050 [0.028, 0.074] | **0.12** | 0.12 |
| mistral:7b probe (k = 1, random) | 126 | 0.056 | 0.050 [0.016, 0.079] | **1.11** | 1.05 |
| llama3.2:3b probe (k = 1, random) | 200 | 0.020 | 0.048 [0.020, 0.080] | **0.42** | 0.39 |

## What this shows

1. **gemma4:12b reads the letter it changes, in every GA seed.** Its repeat rate is 0.04–0.22 of chance in each of the 5 arm-C seeds, every one far below the shuffle range. In the isolated probe it is 0.07 of chance. The prose rule is not ignored by this model; it is followed most of the time.
2. **Adding the fitness line weakened that, in every seed.** gemma's arm F is at 0.31–0.51 of chance against arm C's 0.04–0.22: higher than arm C's highest seed in every F seed. This is the same effect `FITNESS_PROMPT.md` §2 reported as the invalid rate (11.4% vs 4.8%), now expressed against chance. It shows the extra failures are the model reading the letter less, not a change of format.
3. **qwen2.5:7b in the GA behaves close to blind.** The pooled ratio is 1.07 against the shuffle baseline, inside its range, and 1.41 against the marginal baseline. Per seed it ranges from 0.42 to 1.01:
   - in seeds 0 and 3, which have the most repeats (17% and 15% of lines), it is at chance;
   - in seeds 1, 2 and 4 it is below the shuffle range.
   - So qwen avoids the current letter weakly and inconsistently, and in two of five runs not at all.
4. **The same qwen reads the letter in the isolated probe.** At one change per call on random sequences it is at 0.12 of chance, as strong as gemma's 0.07. So **whether the operator reads the sequence is not a fixed property of the model.** For qwen it changes between the probe (one change, random sequence) and the GA (2–4 changes, partly converged sequences). This data cannot separate those two differences.
5. **mistral:7b is at chance even in the probe** (1.11; 7 repeats in 126 lines, so a small sample). **llama3.2:3b** is below chance (0.42), but on only 4 repeats in 200 lines, and its observed rate sits at the bottom edge of the shuffle range; this is weak evidence.

**Reading for the project.** `OPERATOR_BEHAVIOUR.md` found that the operators' *position* choices do not track the letters in the sequence, and that a number in the prompt steers the crossover cut. This measure looks at the other half of a mutate answer, the NEW letter, against the letter it replaces. Here at least one model, gemma, does read the sequence at the position it picked, consistently. So "the operators do not read the genome" is too strong as a general statement. It holds for the position choice and, in the GA, for qwen's letter choice. It does not hold for gemma's letter choice.

## Limits

- **Observational.** Nothing was varied for this measure. The GA data sets differ from the probes in two ways at once (number of changes per call, and random vs converged sequences), and gemma's arms C and F were run ten days apart.
- **The shuffle range is the spread of the chance rate, not a test of the observed rate.** The observed count has its own sampling noise. Seeds whose observed rate is near the edge of the range (qwen seeds 1 and 2) should not be read as decisively below chance.
- **Small probe samples.** The mistral and llama probes are 126 and 200 lines; their ratios are imprecise.
- **One quantity.** This measures whether NEW equals the letter in place. It does not show what else about the sequence the model reads, or whether the letter it picks is any good. The ESM-2 screen (`ESM2_LIKELIHOOD_SCREEN.md`) found that a protein language model's preferences did not predict fitness gain either.
- **The two baselines disagree for qwen** (1.07 vs 1.41). The marginal baseline ignores qwen's strong position preferences and the converged population's letters at those positions; the shuffle baseline keeps both, which is why it is the one used for the readings above.
