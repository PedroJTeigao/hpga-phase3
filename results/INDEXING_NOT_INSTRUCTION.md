# The repeated-letter failure is a position-reading failure, not an instruction-following failure

*A standalone account of one claim, written 2026-10-02. No new model calls were made for this document. Every number comes from `experiments/analyse_position_reading.py`, run over files already committed: the OLD-slot gate in `results/raw/oldfield_gate/` (committed `41ec2b6`, pre-registered `bab55af`, scored in `results/PREREGISTERED_OLDFIELD_SCORED.md`) and the GA call logs of arms C (gemma4:12b and qwen2.5:7b) and F (gemma4:12b). The generated tables are `results/raw/position_reading_tables.md`.*

> **Addendum, 2026-10-02, after the pre-registered intervention test (`results/PREREGISTERED_NUMBERED_SCORED.md`).** The text below is left as written. As a universal claim it was **FALSIFIED** under that test's pre-registered rules. It holds for two of the four models and not for the other two.
>
> - **gemma4:12b and qwen2.5:7b.** Labelling every position in the sequence (`0:A 1:C …`), with the response format unchanged, removed the repeats: 0/640 lines for each in C3. For these two models the failure is a locating failure, now shown by intervention, not only by observation.
> - **mistral:7b and llama3.2:3b.** With the same display they could read the letter (0.79 and 0.96 when asked to state it). Yet in the normal response format they still repeated it at or above chance (5.5–8.2% of lines). Only a response format that made them state the current letter stopped them: llama fell to 0.2% of lines, mistral to 3.1%.
> - So for these two models the failure is not locating but **not consulting** the letter unless the format requires it.
> - **"When a model has stated the letter correctly, it never repeats it" still holds without exception:** 0 of 4,440 more lines.

The mutate operator's format (`mutate/position`) asks for lines `POSITION: <n>, NEW: <letter>`, with a prose rule that NEW must differ from the letter already at that position. Every invalid mutate attempt in this project's GA runs broke that rule (790 of 790 over three arms). The natural reading was that the model ignores the prose rule.

**The data say otherwise.** Across four model families:
- whenever a model correctly states which letter is at the position it chose, it **never** names that letter as the new one;
- every repeat happens on a line where it has **misread** the letter at that position, and those lines repeat the actual letter at about the chance rate;
- reading a letter at a stated index in a space-separated sequence is what fails: often, depending on position for gemma4:12b, and almost everywhere at three changes per call for the three smaller models.

So the failure is in locating the position, not in following the rule.

## Evidence 1. The core table

In the OLD-slot gate each new-format line is `POSITION: <n>, OLD: <the letter currently at that position>, NEW: <letter>`. OLD is the model's own statement of what it read. All conditions and attempts:

| model | lines with OLD stated correctly | of those, NEW = the current letter | lines with OLD stated wrongly | of those, NEW = the actual letter |
|---|---|---|---|---|
| gemma4:12b | 2,176 | **0** | 1,281 | 40 (3.1%) |
| qwen2.5:7b | 704 | **0** | 3,292 | 205 (6.2%) |
| mistral:7b | 339 | **0** | 3,765 | 176 (4.7%) |
| llama3.2:3b | 565 | **0** | 3,950 | 155 (3.9%) |
| **all** | **3,784** | **0** | **12,288** | **576 (4.7%)** |

- 0 of 3,784 correctly read lines repeat the letter.
- 576 of 576 repeats are on misread lines.
- The 4.7% repeat rate on misread lines is about what blind choice gives. The shuffle-chance rate in the gate's cells is 4.2–6.7% (`position_reading_tables.md` §3 and `BLIND_CHOICE.md`'s method).

## Evidence 2. The same pattern without the slot, in the GA's own logs

The OLD slot was a probe; it could itself change behaviour (see the limits). This evidence uses only the production format: the GA's own mutate calls in arm C, first attempts, split by position band. Chance is the marginal rate, from the data set's own NEW-letter distribution.

| data set | positions 0–9 | 10–19 | 20+ |
|---|---|---|---|
| gemma4:12b arm C | 10/1,327 (ratio to chance 0.09) | 5/1,309 (0.05) | **59/1,675 (0.50)** |
| qwen2.5:7b arm C | **0/251 (0.00)** | 160/1,324 (1.54) | 298/2,653 (1.44) |

- **gemma4:12b** repeats almost only at positions 20 and beyond: 59 of its 74 repeats. That is exactly where, in the gate, its reading accuracy drops to 0.23 (Evidence 3).
- **qwen2.5:7b** never repeats at positions 0–9 and repeats above the marginal chance rate beyond position 10.
- This is the same claim measured a second way, independent of the slot. Repeats sit where reading fails.

## Evidence 3. The reading-accuracy profile

Reading accuracy = the share of new-format lines whose OLD equals the letter actually at that position. Blind guessing gives about 0.05.

**gemma4:12b, by position band:**

| condition | 0–9 | 10–19 | 20+ |
|---|---|---|---|
| C1 (1 change, random sequences) | 27/27 = 1.00 | 67/89 = 0.75 | 4/18 = 0.22 |
| C2 (3 changes, random sequences) | 509/510 = 1.00 | 351/489 = 0.72 | 128/561 = 0.23 |
| C3 (2–4 changes, replayed GA sequences) | 542/544 = 1.00 | 393/541 = 0.73 | 155/678 = 0.23 |

- gemma's reading depends on **position**, not on the number of changes or the kind of sequence. The band profile is the same in all three conditions.
- Its overall drop from 0.73 (C1) to 0.62–0.63 (C2, C3) comes only from choosing later positions when asked for more changes.
- Wrong OLD letters are most often the letter one position to the right (`PREREGISTERED_OLDFIELD_SCORED.md`): a counting error, not a guess.

**All four models, by number of changes, within position band:**

| model | 1 change (C1): 0–9 / 10–19 / 20+ | 3 changes (C2): 0–9 / 10–19 / 20+ |
|---|---|---|
| gemma4:12b | 1.00 / 0.75 / 0.22 | 1.00 / 0.72 / 0.23 |
| qwen2.5:7b | **0.88** / 0.20 / 0.05 | **0.25** / 0.20 / 0.12 |
| mistral:7b | 0.27 / 0.12 / 0.06 | 0.11 / 0.09 / 0.07 |
| llama3.2:3b | 0.67 / 0.06 / 0.08 | 0.67 / 0.05 / 0.07 |

- **qwen2.5:7b** reads early positions well with one change and poorly with three, *within the same band*. For qwen, the number of changes requested matters, not just where.
- **llama3.2:3b** reads positions 0–9 at two-thirds and is at chance beyond.
- **mistral:7b** is near chance everywhere.

## Re-reading earlier findings

| earlier finding | status | why |
|---|---|---|
| gemma's modal mutate position is 10 at lengths 40, 63 and 80 (`OPERATOR_BEHAVIOUR.md` Evidence 3) | **unaffected; at most consistent** | 10 lies at the edge of the band gemma reads perfectly (0–9) and inside one it reads at 0.72–0.75. "It picks where it can read" is consistent with that, but does not explain why 10 rather than 0–9. qwen's favourites (0 and 10) argue against it as a general rule: qwen reads 10–19 at only 0.20 even with one change. |
| relabelling the positions 107–169 gave old index 10 zero choices (0/100, both models) | **unaffected** | That result is about which label is chosen. Reading accuracy was never measured under relabelled positions. Nothing here bears on it. |
| "the letter at the chosen position does not predict the choice" (`OPERATOR_BEHAVIOUR.md` Evidence 4) | **partly consistent; not explained for gemma** | For mistral, llama, and qwen beyond position 10, the models mostly cannot locate the letter, so the choice could not depend on it; the null is what that predicts. But Evidence 4 was measured at one change per call, where gemma reads its usual positions (3, 10, 12) correctly 75–100% of the time and still showed no letter dependence. For gemma the null is not explained by misreading. |
| qwen's near-chance repeat ratio in the GA (`BLIND_CHOICE.md`: pooled 1.07 against shuffle chance) | **consistent; partly explained** | Consistent: in the GA, qwen's repeats are absent at 0–9 and all beyond (Evidence 2), where its reading in the gate is poor. Explained in part: `BLIND_CHOICE.md` could not tell whether qwen's probe-vs-GA gap came from the number of changes or from converged sequences. Here, on random sequences only, going from 1 change to 3 raises qwen's old-format repeat ratio from 0.00 to 0.48 and cuts its reading at positions 0–9 from 0.88 to 0.25. So the number of changes is part of the cause. **Not explained:** the remaining gap to the GA's 1.07, and why qwen's GA ratio beyond position 10 is *above* chance (1.44–1.54) rather than at it, which misreading alone would predict. |
| arm F's invalid rate roughly doubling with the fitness line (`FITNESS_PROMPT.md` §2) | **not explained; one explanation ruled out** | If F's extra repeats were extra misreads at far positions, they would sit at 20+. They do not. In F, gemma repeats at positions 0–9 at 0.72 of chance (arm C: 0.09), where it reads almost perfectly in the gate and in arm C. Whatever the fitness line does, it is not only position misreading. The line is inserted directly after the sequence line; its effect on reading was never measured. |

## Limits, stated plainly

1. **The probe-to-GA calibration failed.** On the replayed GA sequences under the production format, the gate drew qwen's failure at 4.1% of lines (5.3% counting retries), against 11.8% in its GA runs. So the gate elicits the failure at less than half the GA rate. Inferences from the gate's reading accuracies to GA behaviour are weak. Evidence 2 is the GA-side support, and it is observational.
2. **The pre-registration did not say which rule takes precedence.** Under `PREREGISTERED_OLDFIELD.md`'s own rules the gate met one falsification condition (qwen's rate rose) and was inconclusive for three of four models (reading accuracy below 50%). Which governs was not specified in advance and is not chosen here. This document's claim rests on a pattern the pre-registration did not anticipate (Evidence 1). It should be read as a strong observational finding, not as a confirmed pre-registered prediction.
3. **The specificity control was weakened.** mistral's wrong-line-count failures vanished under the slot (600 → 0 in C2, 597 → 0 in C3): a general structural effect of a more structured line, not a specific one. A more structured line changed what at least one model did apart from the letter.
4. **The slot may itself change reading.** Reading accuracy was measured with the slot in the prompt. For gemma and qwen it predicts more old-format repeats than occurred, if correct reads never repeat and misreads repeat at chance: gemma C3 predicted about 0.38 of chance against 0.07 observed, qwen C3 about 0.87 against 0.62. So the slot probably made reading worse, and the reading accuracies above are lower bounds for the production format. Evidence 1's conditional result does not depend on this.
5. **The conditional result is observational.** Correctly read lines are not a random sample: they are disproportionately early positions and the stronger models. "0 of 3,784" is a fact about those lines. It does not show what a model would do at a position it could not read if it were given the letter there. That is the next test (below).
6. **Chance baselines differ by section:** shuffle chance in Evidence 1, marginal chance in Evidence 2 (as in `BLIND_CHOICE.md`, where the two disagree for qwen). One prompt family, temperature 0.7, seed 0.

## What this does not show

- **It does not show the operator would search better if indexing were fixed.** Nothing here measures fitness. The project's fitness results stand as they are (`OPERATOR_DOES_NOT_MATTER.md`, `PREREGISTERED_QWEN_SCORED.md`). A model that reads every position perfectly could still choose edits no better than chance; the ESM-2 screen found that even a protein language model's preferences did not predict fitness gain here.
- **It does not show reading the letter is understanding the sequence.** Stating the letter at a position is a retrieval task; nothing here bears on whether a model represents anything about structure.
- **It does not explain the position preferences** (modal 10, the relabelling result), or arm F's increase.
- **It does not establish that format beats wording, or the reverse,** for this failure. The rule was followed whenever it could be; the slot exposed, rather than fixed, the reading failure.
