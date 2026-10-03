# Numbered-display probe: post-hoc descriptive numbers (NOT pre-registered)

*Written 2026-10-02 in answer to two follow-up questions. **These numbers are descriptive and were derived after the fact, outside `results/PREREGISTERED_NUMBERED.md`. They do not revise the verdict in `results/PREREGISTERED_NUMBERED_SCORED.md` (FALSIFIED), and are deliberately not written into that file.** Every count comes from files committed at `b87bee1`. Nothing was re-run. First attempts only, as in the scoring.*

## 1. Unconditional repeat rates in the OLD-slot cells

**What is recorded:**
- Parsed lines and repeats among all of them, per cell, are recorded:
  - `results/raw/numbered_probe/numbered_summary.json`, keys `first.lines` and `first.repeats`;
  - `results/raw/numbered_probe/numbered_tables.md`, columns "lines" and "repeats (rate)", OLD-slot rows at lines 6–36 (even lines).
- The repeat rate on **misread** lines is **not recorded**. It is derived here by arithmetic from recorded counts, as `(repeats − cond_repeats) / (old_lines − old_correct)`.
- That arithmetic is exact. In every OLD-slot cell `first.lines` equals `first.old_lines`, so every parsed line carries an OLD letter, and `cond_repeats` is 0 in every cell.

| model | condition | (a) numbered + OLD: repeats / lines | (b) gate (spaced) + OLD: repeats / lines | (c) numbered + OLD, misread lines only | (c) gate + OLD, misread lines only |
|---|---|---|---|---|---|
| gemma4:12b | C2 | 0 / 600 (0.0%) | 5 / 600 (0.8%) | 0 / 6 | 5 / 230 (2.2%) |
| gemma4:12b | C3 | 0 / 640 (0.0%) | 8 / 640 (1.3%) | 0 / 2 | 8 / 269 (3.0%) |
| qwen2.5:7b | C2 | 8 / 600 (1.3%) | 34 / 600 (5.7%) | 8 / 102 (7.8%) | 34 / 504 (6.7%) |
| qwen2.5:7b | C3 | 14 / 640 (2.2%) | 35 / 640 (5.5%) | 14 / 81 (17.3%) | 35 / 559 (6.3%) |
| mistral:7b | C2 | 20 / 599 (3.3%) | 18 / 600 (3.0%) | 20 / 126 (15.9%) | 18 / 551 (3.3%) |
| mistral:7b | C3 | 20 / 638 (3.1%) | 29 / 640 (4.5%) | 20 / 135 (14.8%) | 29 / 587 (4.9%) |
| llama3.2:3b | C2 | 2 / 600 (0.3%) | 26 / 600 (4.3%) | 2 / 38 (5.3%) | 26 / 545 (4.8%) |
| llama3.2:3b | C3 | 1 / 640 (0.2%) | 27 / 640 (4.2%) | 1 / 27 (3.7%) | 27 / 591 (4.6%) |

**Observations** (descriptive; no test was run):
- **The unconditional rate with the numbered display and OLD slot is 0.0–0.3% for gemma and llama, 1.3–2.2% for qwen and 3.1–3.3% for mistral.** It is not zero for qwen or mistral. All of those repeats are on misread lines.
- **On misread lines, the numbered display goes with higher collision rates for qwen (7.8%, 17.3%) and mistral (15.9%, 14.8%)** than in the gate display (3.3–6.7%), on small denominators (81–135 lines). Why is not known.
- **Numbered + OLD has the lowest unconditional rate of any cell tested only for llama.** For qwen, the numbered display without the slot is lower (0.3% and 0.0%, `numbered_tables.md` base rows). For mistral, numbered + OLD (3.1–3.3%) is no lower than the gate display with the slot in C2 (3.0%).

## 2. mistral's denominator of 441

**How it is produced:**
- Per cell, never matched across cells. `experiments/probe_mutate_numbered.py` at `b87bee1`, `score_file`:
  - **line 200:** `c["lines"] += 1` for every parsed `POSITION, NEW` line of a first attempt;
  - **line 198:** a line is skipped only if its position lies beyond the sequence.
- **No cap, no min across cells, no truncation.** Lines from attempts the production parser rejected are counted too, as the pre-registration defines "parsed lines".
- **Nothing is discarded**, so there is no selection to bias the rate.

**Why both C3 cells are exactly 441:** counted from the committed raw call logs, first attempts. This is computed after the fact.

| cell | source | requests by k | lines requested | lines returned | (returned − requested) per call |
|---|---|---|---|---|---|
| C2 spaced/base | `oldfield_gate/calls_mistral_7b_C2_old.jsonl` | k=3: 200 | 600 | 400 | −1 in 200/200 |
| C2 numbered/base | `numbered_probe/calls_mistral_7b_C2_num.jsonl` | k=3: 200 | 600 | 400 | −1 in 200/200 |
| C3 spaced/base | `oldfield_gate/calls_mistral_7b_C3_old.jsonl` | k=2: 1, k=3: 158, k=4: 41 | 640 | 441 | −1 in 199/200, 0 in 1/200 |
| C3 numbered/base | `numbered_probe/calls_mistral_7b_C3_num.jsonl` | k=2: 1, k=3: 158, k=4: 41 | 640 | 441 | −1 in 199/200, 0 in 1/200 |

- mistral returned exactly one line fewer than requested in **every** call with k = 3 or 4, in both displays.
- The one exception in each C3 cell is the **same call**: input 137, the only input with k = 2. There it returned both lines (spaced: `POSITION: 1 … / POSITION: 18 …`; numbered: `POSITION: 6 … / POSITION: 11 …`).
- So 640 − 199 = 441 in both cells. The coincidence is structural, not chance: the two cells share inputs, and mistral's line count depends only on k.
- **Calls returning 2 lines instead of 3:** C2, 200/200 in both cells. C3, 158/158 of the k = 3 calls in both cells; the 41 k = 4 calls each returned 3 lines.

**"Is the dropped line systematically the last one requested?" Not answerable from the files.**
- The prompt asks for "exactly k position(s)" and leaves the positions to the model.
- No specific lines or positions are requested, so there is no "last requested line" to compare against.

**Consequence already stated in the scoring file's Limits:** mistral's base-format repeat rates rest on 400–441 lines, not 600–640. All of mistral's first attempts in those cells are invalid for line count. This explains the denominator and changes nothing in the verdict.
