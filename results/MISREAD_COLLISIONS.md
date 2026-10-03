*Post hoc and descriptive: written 2026-10-03 from logs committed at `b87bee1`. Nothing here was pre-registered. No probe was re-run, and no pre-registered verdict is changed (`PREREGISTERED_OLDFIELD_SCORED.md`, `PREREGISTERED_NUMBERED_SCORED.md`).*

# On misread lines, is the replacement letter chosen independently of the true letter?

## The question

In the OLD-slot cells, every repeat of the current letter is on a line the model misread: its stated OLD differed from the true letter at that position (`INDEXING_NOT_INSTRUCTION.md`, `PREREGISTERED_NUMBERED_SCORED.md`). So the unconditional repeat rate is exactly P(misread) × P(repeat | misread).

This file asks whether P(repeat | misread), the rate at which NEW equals the true letter on a misread line, is at chance.

## Method

- **Data:** first attempts of the OLD-slot cells, C2 and C3, four models, two displays.
  - **spaced:** the OLD-slot gate's new-format cells, `results/raw/oldfield_gate/calls_<model>_<C2|C3>_new.jsonl`;
  - **numbered:** the numbered probe's numbered+OLD cells, `results/raw/numbered_probe/calls_<model>_<C2|C3>_numold.jsonl`.
- **Misread lines:** OLD ≠ the true letter at that position. Correctly read lines are excluded; they contribute 0 repeats anyway. Attempts with no misread line are dropped.
- **Baselines:** `measure()` from `experiments/analyse_blind_choice.py`, **unchanged**, with each cell's misread lines as the data set:
  - **shuffle:** the cell's (position, NEW) lines re-paired with other attempts' sequences from the same cell, 200 pairings.
  - **marginal:** the cell's own NEW-letter distribution.
- **Which baseline is primary.** The script computes both and does not designate a baseline as primary in code; its "Primary" refers to first attempts. `BLIND_CHOICE.md` calls shuffle "the stricter baseline, and the one used for the readings". It is listed first here, and the readings below use it, with marginal alongside.
- **Test:** two-sided exact binomial of the observed repeat count against n lines at each baseline's expected rate. **15 defined cells per baseline, uncorrected.** A Bonferroni threshold over 15 would be p < 0.0033.
- **Code:** `experiments/analyse_misread_collisions.py`. Tables: `results/raw/misread_collisions_tables.md` and `results/raw/misread_collisions.json`.

## Results

| model | cond | display | n misread lines | observed P(NEW = true letter) | shuffle expected [95% range] | ratio, p (shuffle) | marginal expected | ratio, p (marginal) |
|---|---|---|---|---|---|---|---|---|
| gemma4:12b | C2 | spaced | 230 | 5 (0.022) | 0.050 [0.030, 0.078] | 0.44, p = 0.048 | 0.051 | 0.42, p = 0.036 |
| gemma4:12b | C2 | numbered | 6 | 0 (0.000) | 0.029 [0.000, 0.167] | 0.00, p = 1 | 0.083 | 0.00, p = 1 |
| gemma4:12b | C3 | spaced | 269 | 8 (0.030) | 0.062 [0.038, 0.096] | 0.48, p = 0.023 | 0.055 | 0.55, p = 0.080 |
| gemma4:12b | C3 | numbered | 2 | 0 | undefined: marginal expectation 0, too few lines | | | |
| qwen2.5:7b | C2 | spaced | 504 | 34 (0.067) | 0.052 [0.034, 0.069] | 1.30, p = 0.13 | 0.053 | 1.28, p = 0.16 |
| qwen2.5:7b | C2 | numbered | 102 | 8 (0.078) | 0.057 [0.020, 0.098] | 1.38, p = 0.39 | 0.048 | 1.62, p = 0.16 |
| qwen2.5:7b | C3 | spaced | 559 | 35 (0.063) | 0.043 [0.027, 0.062] | 1.47, p = 0.027 | 0.049 | 1.28, p = 0.14 |
| qwen2.5:7b | C3 | numbered | 81 | 14 (0.173) | 0.080 [0.027, 0.143] | **2.15, p = 0.0061** | 0.052 | **3.33, p = 7.0e-05** |
| mistral:7b | C2 | spaced | 551 | 18 (0.033) | 0.051 [0.031, 0.069] | 0.63, p = 0.043 | 0.048 | 0.67, p = 0.091 |
| mistral:7b | C2 | numbered | 126 | 20 (0.159) | 0.052 [0.016, 0.087] | **3.04, p = 8.8e-06** | 0.051 | **3.10, p = 6.8e-06** |
| mistral:7b | C3 | spaced | 587 | 29 (0.049) | 0.057 [0.040, 0.075] | 0.87, p = 0.53 | 0.053 | 0.93, p = 0.78 |
| mistral:7b | C3 | numbered | 135 | 20 (0.148) | 0.066 [0.030, 0.106] | **2.24, p = 0.00070** | 0.050 | **2.98, p = 1.3e-05** |
| llama3.2:3b | C2 | spaced | 545 | 26 (0.048) | 0.047 [0.031, 0.066] | 1.00, p = 0.92 | 0.043 | 1.10, p = 0.60 |
| llama3.2:3b | C2 | numbered | 38 | 2 (0.053) | 0.040 [0.000, 0.105] | 1.31, p = 0.67 | 0.044 | 1.19, p = 0.69 |
| llama3.2:3b | C3 | spaced | 591 | 27 (0.046) | 0.048 [0.032, 0.065] | 0.96, p = 0.92 | 0.045 | 1.00, p = 0.92 |
| llama3.2:3b | C3 | numbered | 27 | 1 (0.037) | 0.036 [0.000, 0.111] | 1.03, p = 1 | 0.052 | 0.71, p = 1 |

## Readings

**Spaced display: at chance for llama, and for mistral in C3; not uniformly at chance.**
- llama: ratio 1.00 (p = 0.92) and 0.96 (p = 0.92) against shuffle.
- mistral, C3: 0.87 (p = 0.53).
- For those cells, "on misread lines the replacement letter is chosen independently of the true letter" is supported.
- It is **not** supported as a general statement:
  - gemma is nominally below chance in both conditions: 0.44 (p = 0.048) and 0.48 (p = 0.023) against shuffle;
  - qwen in C3 is nominally above: 1.47 (p = 0.027);
  - mistral in C2 is nominally below: 0.63 (p = 0.043).
- None of these nominal deviations survives a Bonferroni correction over the 15 cells (p < 0.0033). The back-of-envelope clustering near 1/19 is therefore mostly, but not exactly, chance on the proper baseline.

**Numbered display: qwen's 17.3% and mistral's 14.8–15.9% are above chance on both baselines. That is unexplained.**
- qwen, C3: 2.15x shuffle (p = 0.0061) and 3.33x marginal (p = 7.0e-05).
- mistral, C2: 3.04x shuffle (p = 8.8e-06).
- mistral, C3: 2.24x shuffle (p = 0.00070).
- All except qwen-C3-against-shuffle also clear the Bonferroni threshold.
- qwen's 7.8% in C2 is not distinguishable from chance (1.38, p = 0.39).
- The proper baseline does not absorb these. No mechanism is proposed here.

## Limits

- **Post hoc.** The misread subset is defined by the model's own OLD answer. So it is a selected set of lines, and it differs in composition between displays: far fewer lines are misread in the numbered display (27–135 for three models, 2–6 for gemma, against 230–591 spaced).
- **Small numbered-display cells.** gemma's two numbered cells (6 and 2 lines) are uninformative; one has no defined baseline.
- **One prompt family, temperature 0.7, seed 0.** Single runs of 200 calls per cell.
- **No new calls.** Every number can be regenerated from the committed logs with the script named above.
