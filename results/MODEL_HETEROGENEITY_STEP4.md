# Model heterogeneity, step 4: absolute or relative position bias, and does the chosen position depend on the letter there?

*Branch `model-heterogeneity`. No GA, no fitness. Part 1 is new calls (mutate/position at three genome lengths). Part 2 is offline, from call logs that already existed; no new calls. Reported separately. Scripts: `experiments/run_mutate_length_sweep.py`, `experiments/summarize_mutate_length_sweep.py`, `experiments/analyze_position_vs_letter.py`. Tables are the generated files `results/raw/model_lensweep_tables.md` and `results/raw/model_position_vs_letter_tables.md` (the latter also holds the per-letter counts, omitted below). Raw: `sequence_operator_compliance_<model>_len<N>_lensweep.json`, `llm_operator_calls_lensweep_len<N>_<model>.jsonl`, console and driver logs.*

## 1. Genome-length sweep for mutate/position

**Setup.** `gemma4:12b` and `qwen2.5:7b`, 100 `mutate/position` calls at genome lengths 40, 63 and 80 (the operator bounds are [30, 80]), temperature 0.7, seed 0, `num_ctx` 4096, run with the existing probe (unchanged, `--genome-len`), which sets the mutation rate to 1/N so k = 1 at every length. Each cell is a fresh run with the RNG restarted from seed 0. Question: does the modal position stay at the same absolute index (a salient-index effect) or move with the length (a relative-position effect)?

### Results (generated)

100 mutate/position calls per cell, temperature 0.7, seed 0, one change per call (k = 1). Each model reported on its own.

## gemma4:12b

| genome length N | fell back | requests/call | successful calls | distinct positions (of N; expected if uniform) | modal position (share) | modal position / (N-1) | top-3 share | chi2 vs uniform (df N-1), MC p | choices at multiples of 10 (share; uniform would give) |
|---|---|---|---|---|---|---|---|---|---|
| 40 | 0/100 | 1.01 | 100 | 13 (36.8) | 10 (38.0%) | 0.256 | 70.0% | 785, p=5e-05 | 39.0% (10.0%) |
| 63 | 0/100 | 1.00 | 100 | 17 (50.3) | 10 (38.0%) | 0.161 | 62.0% | 1083, p=5e-05 | 41.0% (11.1%) |
| 80 | 0/100 | 1.00 | 100 | 13 (57.3) | 10 (42.0%) | 0.127 | 64.0% | 1662, p=5e-05 | 44.0% (10.0%) |

Full distribution (position x count), most used first:

- **N=40**: 10x38, 3x24, 2x8, 12x7, 15x6, 1x4, 13x3, 4x2, 11x2, 14x2, 18x2, 0x1, 5x1
- **N=63**: 10x38, 3x12, 12x12, 14x6, 11x5, 15x5, 4x4, 5x4, 2x3, 21x3, 22x2, 0x1, 1x1, 13x1, 20x1, 30x1, 32x1
- **N=80**: 10x42, 3x11, 12x11, 15x9, 7x7, 4x5, 14x4, 2x3, 11x3, 20x2, 5x1, 31x1, 34x1

Predictions from the N=63 mode (10): absolute hypothesis -> the same index at N=40 and N=80 (10, 10); relative hypothesis -> 6.3 at N=40 and 12.7 at N=80. Observed modal positions: N=40: 10, N=63: 10, N=80: 10.

Replication at N=63: step 1 position counts identical to this run.

## qwen2.5:7b

| genome length N | fell back | requests/call | successful calls | distinct positions (of N; expected if uniform) | modal position (share) | modal position / (N-1) | top-3 share | chi2 vs uniform (df N-1), MC p | choices at multiples of 10 (share; uniform would give) |
|---|---|---|---|---|---|---|---|---|---|
| 40 | 0/100 | 1.25 | 100 | 4 (36.8) | 0 (54.0%) | 0.000 | 99.0% | 1506, p=5e-05 | 83.0% (10.0%) |
| 63 | 0/100 | 1.03 | 100 | 6 (50.3) | 10 (49.0%) | 0.161 | 83.0% | 1926, p=5e-05 | 74.0% (11.1%) |
| 80 | 0/100 | 1.01 | 100 | 5 (57.3) | 10 (67.0%) | 0.127 | 97.0% | 4126, p=5e-05 | 97.0% (10.0%) |

Full distribution (position x count), most used first:

- **N=40**: 0x54, 10x29, 1x16, 4x1
- **N=63**: 10x49, 0x25, 2x9, 1x8, 3x6, 12x3
- **N=80**: 10x67, 0x28, 4x2, 20x2, 3x1

Predictions from the N=63 mode (10): absolute hypothesis -> the same index at N=40 and N=80 (10, 10); relative hypothesis -> 6.3 at N=40 and 12.7 at N=80. Observed modal positions: N=40: 0, N=63: 10, N=80: 10.

Replication at N=63: step 1 position counts DIFFERENT from this run.


### Answer: absolute, not relative

**The favoured positions stay at the same absolute indices across lengths; they do not scale with length.**

- `gemma4:12b`: the mode is position 10 at every length (38%, 38%, 42% of choices at N = 40, 63, 80). A relative-position effect would have put it at about 6.3 at N=40 and 12.7 at N=80 (the N=63 mode scaled by length); it did not move. The second and third favourites also keep their absolute values (3 and 12 at N=63 and N=80; 3 at N=40).
- `qwen2.5:7b`: the two favourites are positions 0 and 10 at every length (0 and 10 take 83%, 74% and 95% of choices at N = 40, 63, 80), again fixed indices. Which of the two is on top changes with length: 0 leads at N=40 (54% vs 29% for position 10), 10 leads at N=63 (49% vs 25%) and N=80 (67% vs 28%). So qwen2.5:7b's modal position is not constant, but its favourite indices are, and neither moves proportionally with N.
- Choices landing on a multiple of 10 are 39% to 44% for gemma4:12b and 74% to 97% for qwen2.5:7b, against 10% to 11% if positions were uniform.

**What this does not separate.** "Round number" and "small index near the start" both fit. Every favourite is a low index: at N=80 no position above 34 was chosen by gemma4:12b and none above 20 by qwen2.5:7b, and several favourites are not round (gemma4:12b's 3, 12, 15; qwen2.5:7b's 1, 2, 3). Position 10 dominating is round-number-like, and 0 for qwen2.5:7b is too, but the sweep cannot tell a taste for round numbers from a bias toward early positions. Testing that would need a prompt that moves the "start" (for example a leading pad) or a position range that does not start at 0, which was not done.

**Reproducibility caveat.** At N=63 the length-sweep run and step 1 used the same seed and prompts. `gemma4:12b` reproduced step 1 exactly. `qwen2.5:7b` did not: the position counts differ (step 1: 0×11, 1×9, 2×12, 3×11, 10×55, 12×2; sweep: 0×25, 1×8, 2×9, 3×6, 10×49, 12×3), because a retry at call 49 of the sweep run (3 retries in all, versus none in step 1) shifted the RNG stream for the remaining calls: each request attempt draws its sampling seed from the shared RNG (`hpga/operators.py` line 463), so an extra attempt changes every later genome and seed. The set of positions and the mode (10) are the same in both; the count at position 0 differs by 14 (11 vs 25). So run-to-run variation in a single position's count can be that large at n=100, which is smaller than the N=40 margin (0 leads position 10 by 25 calls) but comparable to the gaps between qwen2.5:7b's minor positions at N=63.

**Other limits.** One seed, 100 calls per cell, two models; lengths 40, 63, 80 only.

## 2. Offline: is the chosen position predicted by the letter that sits there?

**Method (no model calls).** From the existing `mutate/position` call logs: the four step-1 logs (100 calls each, genome length 63; llama3.2:3b has 59 successful calls, mistral:7b 96) and the earlier `gemma4:12b` length-63 run behind PHASE3_RESULTS.md §9.2 (400 calls, `llm_operator_calls_1789749658_409662.jsonl`). Each call used a fresh uniform-random genome. For every call I read the genome from the prompt, the chosen position from the first valid response, and the letter sitting at that position; calls that fell back are excluded. The letters at chosen positions are compared with the letters available in the same genomes, by Monte Carlo chi-square (20,000 draws) under two nulls: **A**, position uniform over the genome (the plain comparison against the background), and **B**, the model's own chosen positions kept but paired with genomes from other calls, which preserves the positional bias and breaks any link to content. Rejecting B would mean the choice depends on the letter. The same tests are run on four residue classes (hydrophobic AVILMFW, charged DEKRH, polar STNQCY, special GP).

### Results (generated)

Offline: is the chosen position predicted by the letter that sits there? (no model calls; calls that fell back are excluded)

| source | calls used | letters: chi2 (df 19) vs A (uniform position), MC p | letters: chi2 vs B (positions kept, content shuffled), MC p | classes: chi2 (df 3) vs A, p | classes: chi2 vs B, p |
|---|---|---|---|---|---|
| gemma4:12b (step 1, n=100) | 100 | 10.9, p=0.923 | 12.1, p=0.740 | 1.07, p=0.787 | 0.39, p=0.924 |
| gemma4:12b (P3 sec. 9 run, n=400) | 400 | 29.0, p=0.058 | 25.7, p=0.035 | 2.21, p=0.524 | 1.49, p=0.600 |
| llama3.2:3b (step 1) | 59 | 22.8, p=0.229 | 20.9, p=0.238 | 1.12, p=0.769 | 1.79, p=0.591 |
| qwen2.5:7b (step 1) | 100 | 19.0, p=0.433 | 9.9, p=0.718 | 4.15, p=0.235 | 2.13, p=0.335 |
| mistral:7b (step 1) | 96 | 10.9, p=0.922 | 10.8, p=0.920 | 1.59, p=0.659 | 1.03, p=0.795 |

Monte Carlo p-values, 20000 draws each (floor 5.0e-05). Null A = position uniform over the genome; null B = the model's own positions kept but paired with genomes from other calls. Expected count per letter is ~5 at n=100 and ~20 at n=400, so a modest content preference would not be detectable in the n=100 rows.

### Residue classes at the chosen position

| source | class | share of all genome letters | share at chosen positions | expected share under B |
|---|---|---|---|---|
| gemma4:12b (step 1, n=100) | hydrophobic | 35.2% | 36.0% | 36.0% |
| gemma4:12b (step 1, n=100) | charged | 24.7% | 23.0% | 25.3% |
| gemma4:12b (step 1, n=100) | polar | 30.0% | 28.0% | 27.2% |
| gemma4:12b (step 1, n=100) | special | 10.1% | 13.0% | 11.6% |
| gemma4:12b (P3 sec. 9 run, n=400) | hydrophobic | 35.2% | 36.8% | 35.0% |
| gemma4:12b (P3 sec. 9 run, n=400) | charged | 24.8% | 24.2% | 25.5% |
| gemma4:12b (P3 sec. 9 run, n=400) | polar | 29.9% | 27.2% | 28.9% |
| gemma4:12b (P3 sec. 9 run, n=400) | special | 10.2% | 11.8% | 10.6% |
| llama3.2:3b (step 1) | hydrophobic | 35.7% | 35.6% | 35.6% |
| llama3.2:3b (step 1) | charged | 24.5% | 23.7% | 23.9% |
| llama3.2:3b (step 1) | polar | 30.1% | 27.1% | 31.6% |
| llama3.2:3b (step 1) | special | 9.7% | 13.6% | 8.9% |
| qwen2.5:7b (step 1) | hydrophobic | 35.2% | 42.0% | 36.3% |
| qwen2.5:7b (step 1) | charged | 24.7% | 26.0% | 25.0% |
| qwen2.5:7b (step 1) | polar | 30.0% | 21.0% | 25.9% |
| qwen2.5:7b (step 1) | special | 10.1% | 11.0% | 12.9% |
| mistral:7b (step 1) | hydrophobic | 35.0% | 34.4% | 34.3% |
| mistral:7b (step 1) | charged | 25.0% | 30.2% | 26.1% |
| mistral:7b (step 1) | polar | 30.0% | 26.0% | 29.2% |
| mistral:7b (step 1) | special | 10.0% | 9.4% | 10.4% |


### Answer: no detectable dependence on the letter

For every model, the letters at the chosen positions are statistically indistinguishable from the background, and from the content-shuffled null:

- Letter level, null B: p = 0.74 (gemma4:12b, n=100), 0.24 (llama3.2:3b), 0.72 (qwen2.5:7b), 0.92 (mistral:7b), and 0.035 for gemma4:12b at n=400. Null A gives p = 0.92, 0.23, 0.43, 0.92 and 0.058.
- Class level: every p is at least 0.23 under either null (0.60 under B for gemma4:12b at n=400). For example, hydrophobic residues make up 35.2% of the available letters and 36.8% of the letters at gemma4:12b's 400 chosen positions (35.0% expected under B).
- The one nominal p below 0.05 (gemma4:12b, n=400, letter level, null B, p = 0.035) is one of ten letter-level tests and is not corrected for that; its pattern is not coherent (C and T chosen 10 times each against ~18 expected, N, V and W chosen 26 to 28 times against ~17 to 20), and the class-level test on the same data shows nothing. I do not read it as content sensitivity.

So the position is chosen without regard to which letter sits there, to the resolution of these tests: the operator is behaving as if it ignores the content. Two limits on how strongly to say that. At about 5 expected calls per letter (n=100) only large letter-specific preferences are detectable; the class-level tests (4 cells) and gemma4:12b at n=400 are the most sensitive and show nothing. And this tests only the letter at the chosen position; it does not test other kinds of content sensitivity (neighbouring letters, local motifs, or the letter it chooses to put in).
