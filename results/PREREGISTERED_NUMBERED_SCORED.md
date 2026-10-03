# Scoring the numbered-display probe against its pre-registration

*This scores `results/PREREGISTERED_NUMBERED.md`, committed as `038cdf8` on 2026-10-02 before any call with the numbered display. That file is not edited. Code `99e0928`; GPU-check fix `af74fb9`, committed before the second launch. Runs 2026-10-02: gemma4:12b 13:34–15:34, then qwen2.5:7b, mistral:7b and llama3.2:3b 16:10–20:39. Raw data, `numbered_tables.md`, `numbered_summary.json` and the logs are in `results/raw/numbered_probe/`. Scope: first attempts only, as pre-registered.*

## Run notes

- **Two launches.** The first stopped after gemma on a false alarm: our own exiting Ollama runner, which `nvidia-smi` names `[No data]`. That is the same cause as the OLD-slot gate's three stops (see the dated correction in `PREREGISTERED_OLDFIELD_SCORED.md`).
- The check was fixed (`af74fb9`) and tested live before the second launch, which ran all three remaining models without a stop.
- **No measurement was affected.** The stop fell between models.

## The rules, applied in the pre-registered order

**Rule 1, drift.** Every model's spaced-display C1 prompts reproduce the gate's byte for byte: gemma 100/100, qwen 100/100, mistral 138/138, llama 266/266. The gate's spaced cells are reused for all four.

**Rule 2, delivery** (numbered + OLD-slot reading accuracy in C3 ≥ 0.50):

| model | reading in C3 | delivered |
|---|---|---|
| gemma4:12b | 1.00 | yes |
| qwen2.5:7b | 0.87 | yes |
| mistral:7b | 0.79 | yes |
| llama3.2:3b | 0.96 | yes |

**Rule 3, falsification**, over delivered models.

**(a)** Each delivered model with ≥ 10 repeated lines in spaced/base C3, comparing its repeat rate with the numbered display and normal response (numbered/base):

| model | spaced/base C3 | numbered/base C3 | change | falls by ≥ 50%? |
|---|---|---|---|---|
| gemma4:12b | 3/640 (0.5%), below 10 lines: not counted | 0/640 | | |
| qwen2.5:7b | 26/640 (4.1%) | 0/640 (0.0%) | −100% | yes |
| mistral:7b | 24/441 (5.4%) | 36/441 (8.2%) | **+51%** | **no** |
| llama3.2:3b | 32/640 (5.0%) | 35/640 (5.5%) | **+10%** | **no** |

**(a) is met, by mistral and by llama.**

**(b)** In the numbered + OLD cells, repeats among lines where OLD was correct:

| model | C2 | C3 |
|---|---|---|
| gemma4:12b | 0/594 | 0/638 |
| qwen2.5:7b | 0/498 | 0/559 |
| mistral:7b | 0/473 | 0/503 |
| llama3.2:3b | 0/562 | 0/613 |

**(b) is not met:** 0 of 4,440.

**Verdict: FALSIFIED**, by rule 3(a). Under the pre-registered order, rule 4 (confirmation) is not consulted.

## Every prediction

| model | prediction | measured | score |
|---|---|---|---|
| gemma4:12b | numbered/base C3 ≤ 0.5% | 0/640 | TRUE |
| gemma4:12b | numbered + OLD reading ≥ 0.95 in every band | C3 1.00 / 1.00 / 0.99; C2 1.00 / 1.00 / 0.98 (bands 0–9 / 10–19 / 20+) | TRUE |
| gemma4:12b | band profile flat (20+ within 0.05 of 0–9) | 0.99 vs 1.00 (C3); 0.98 vs 1.00 (C2) | TRUE |
| qwen2.5:7b | numbered/base C3 ≤ 1.0% | 0/640 | TRUE |
| qwen2.5:7b | numbered + OLD reading ≥ 0.80 | 0.87 (C3), 0.83 (C2) | TRUE |
| mistral:7b | numbered/base C3 ≤ 1.5% | 8.2% | **FALSE** |
| mistral:7b | numbered + OLD reading ≥ 0.50 | 0.79 (C3 and C2) | TRUE |
| llama3.2:3b | numbered/base C3 ≤ 1.5% | 5.5% | **FALSE** |
| llama3.2:3b | numbered + OLD reading ≥ 0.70 | 0.96 (C3), 0.94 (C2) | TRUE |
| all | C2 in the same direction as C3 | gemma 0.8% → 0.3%; qwen 2.3% → 0.3%; mistral 4.3% → 5.2%; llama 4.0% → 5.5% | TRUE for gemma and qwen, FALSE for mistral and llama |
| all | conditional repeat rate ≤ 0.5% | 0 of 4,440 | TRUE |

## What the falsification shows

The pre-registered claim was universal: "the repeated-letter failure is a reading failure; make locating easy and it disappears". It holds for two models and fails for two.

To see why, here is each cell's repeat rate against shuffle chance (`BLIND_CHOICE.md`'s method; first attempts):

| model | condition | spaced/base | numbered/base | numbered + OLD |
|---|---|---|---|---|
| gemma4:12b | C2 / C3 | 0.18 / 0.07 | 0.06 / **0.00** | 0.00 / 0.00 |
| qwen2.5:7b | C2 / C3 | 0.47 / 0.63 | 0.07 / **0.00** | 0.26 / 0.38 |
| mistral:7b | C2 / C3 | 0.85 / 0.89 | **1.04 / 1.42** | 0.65 / 0.55 |
| llama3.2:3b | C2 / C3 | 0.84 / 1.13 | **1.19 / 1.30** | 0.09 / **0.04** |

1. **gemma and qwen: the failure was a locating failure, now shown by intervention.** With every position labelled and the response unchanged, both stop repeating the letter: 0 of 640 in C3, and 2 of 600 in C2. For these two models the claim of `INDEXING_NOT_INSTRUCTION.md` holds, and no longer rests only on observation.
2. **mistral and llama: locating is not enough. They also have to be made to look.**
   - With the numbered display they can read the letter at a position: 0.79–0.96 when asked to state it.
   - In the normal response format they still repeat it at, or above, chance.
   - Only when the response requires them to state the current letter do they avoid it: llama falls to 0.002–0.003 of lines (0.04–0.09 of chance), mistral to 0.031–0.033 (0.55–0.65).
   - For these models the failure is that, in the production format, **they do not consult the letter at the position they choose**, even when locating it is easy.
3. **The format result returns, in a narrower form.** The OLD-slot gate failed to fix the repeats because the models could not locate. With locating made easy, the same slot nearly eliminates them for llama and halves them for mistral. For the models that do not look on their own, a response format that makes them state the letter works where the prose rule does not, but only once locating is possible.
4. **The conditional fact holds without exception.** Across 4,440 more correctly read lines in four models, no model ever named, as its NEW letter, the letter it had just stated as current (with the OLD-slot gate: 0 of 3,784 there, all attempts). When a model has the letter in front of it, explicitly, the rule is obeyed.

## A gap in the pre-registration, stated plainly

- Rule 2 measured delivery in the numbered + OLD cell. Rule 3(a) then tested repeats in the numbered/base cell. The design assumed that if a model *can* read with the numbered display, it *does* read when not asked to.
- mistral and llama show that assumption was wrong. It is why the falsification lands on them, and it is why the result is a model split rather than a clean negative.
- This is noted, not used to change the verdict. The verdict under the pre-registered rules is FALSIFIED.

## Other observations (not scored)

- **Cost.** gemma 2 h 0 m; qwen 1 h 16 m; mistral 2 h 31 m; llama 41 m.
  - Retries came almost entirely from the OLD-slot cells: qwen 1.5–1.7 requests per call, mistral 1.8–1.9, llama 1.7–1.8.
  - mistral's numbered/base cells kept failing on line count (200/200 and 199/200 fallbacks; it answers 2 lines when asked for 3), as in the gate.
- **qwen with the numbered display repeats more with the OLD slot than without it** (1.3–2.2% vs 0.0–0.3% of lines). Its reading is imperfect (0.83–0.87), and the misread lines collide.
- **mistral's numbered/base ratio is above chance in C3 (1.42).** Not explained.
  - Cross-reference, added 2026-10-03: post-hoc, descriptive collision rates on misread lines are in `results/MISREAD_COLLISIONS.md`; they do not revise this verdict.

## Limits

- Isolated probe. One prompt family, temperature 0.7, seed 0. 200 calls per cell, a single run each.
- No GA, no fitness. Nothing here bears on search quality.
- The numbered display makes prompts about 3x longer in sequence tokens, which may itself change behaviour.
- mistral answers 2 lines when asked for 3 in both base cells, so its base-format numbers rest on about 400–440 lines, not 600–640.
