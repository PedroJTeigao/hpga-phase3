# Pre-registered predictions: moving the "new letter must differ" rule from prose into the response format

**This file is a prediction, not a result.** It was written on 2026-10-01, before any call with the new format was made. It is committed before the first call, and not edited afterwards. Outcomes will be scored in a separate file that cites this one.

## The question

- Every invalid `mutate/position` attempt in the project's GA runs was the same failure: the model named a NEW letter equal to the letter already at that position. That was 73 of 73 in gemma arm C, 103 of 103 in gemma arm F and 614 of 614 in qwen arm C.
- The rule that forbids it is stated only in prose: "the new letter MUST differ from whatever letter is currently at that position".
- The project's operator findings say that output format decides compliance where wording did not (`OPERATOR_BEHAVIOUR.md` Evidence 1).
- The test: add a slot to the response format that makes the model state the current letter, and see whether the failure disappears.

## What is already known, and constrains the predictions

1. **The failure is specific to the GA's 2–4 changes per call. It is not the probes' failure.**
   - The model-heterogeneity probe (`MODEL_HETEROGENEITY_STEP1.md`) asked for exactly 1 change per call (k = 1) on random sequences.
   - There, gemma4:12b and qwen2.5:7b made 0 invalid attempts in 100 calls.
   - mistral:7b made 42 invalid attempts in 138: 38 of them gave the wrong number of lines, and only 4 were the repeated letter.
   - llama3.2:3b made 207 invalid attempts in 266: 206 gave the wrong number of lines, and only 1 was the repeated letter.
   - At k = 2–4 on random sequences (the circle-proposal probe, gemma), 3 of 52 attempts were the repeated letter.
2. **gemma already follows the prose rule, mostly. qwen does not.** Per line, against chance, in the GA logs:
   - "Chance" assumes the model chooses its NEW letter from its own letter preferences, without looking at the letter in place.
   - **gemma arm C:** 1.8% repeated, against a chance rate of 7.5%. That is 0.23x chance: gemma reads the current letter and avoids it about three times in four.
   - **gemma arm F:** 4.6% repeated, against 8.6% chance (0.54x).
   - **qwen arm C:** 11.8% repeated, against 8.0% chance (1.49x), or 10.8% by shuffling sequences between calls (about 1.1x). qwen behaves as if it never consults the current letter.
   - So "these operators ignore prose" is true for qwen and only partly true for gemma.
   - The test is therefore whether a format slot does reliably what the prose sentence does partly for gemma and not at all for qwen.

## The change, and the design

**New style `position_old`.** It is identical to the current `mutate/position` prompt except for one line: the response-format line gains an OLD slot.

```
current:  POSITION: <0-62>, NEW: <one letter from {A,...,Y}>
new:      POSITION: <0-62>, OLD: <the letter currently at that position>, NEW: <one letter from {A,...,Y}>
```

- Every other byte is unchanged: the instruction prose including the "MUST differ" sentence, the system prompt and the retry hint.
- One necessary difference: `num_predict` rises from `max(24, 10·k)` to `max(48, 16·k)`, because each line is about 4 tokens longer. With the old limit, a k = 3 answer would be cut off and scored invalid for reasons unrelated to the rule.
- **Parsing.** Each line's OLD is compared with the actual letter. Two validity rules are reported:
  - **strict:** OLD must be correct and NEW must differ;
  - **lenient:** OLD is ignored, which is the current rule.
- **Primary metric** is defined identically under both formats: the share of parsed lines whose NEW equals the actual letter at that position (the repeated-letter rate per line). Also reported: invalid-attempt rate (strict and lenient), share of invalid attempts by kind, fallback rate, and OLD accuracy.

**Conditions.** Each model runs every condition under the old and the new format. Temperature 0.7, `num_ctx` 4096, retries 2, seed 0, and the same inputs for both formats.

| condition | inputs | k | calls per format per model |
|---|---|---|---|
| C1, replication | random sequences of length 63, the step-1 settings | 1 | 100 |
| C2, random, GA-sized k | random sequences of length 63 | 3 (rate 0.05) | 200 |
| C3, replay | 200 real GA base sequences with their own k (2–4): 100 drawn from gemma arm C's mutate prompts and 100 from qwen arm C's, seeded, the same 200 for every model | 2–4 | 200 |

- C3 is the condition where the failure is known to occur.
- Under the old format, qwen on C3 is a calibration check: it should reproduce something near its GA rate (11.8% of lines). If it does not, the isolated probe does not reproduce the GA failure, and the conclusions are limited to the probe.

## Predictions

Rates are per parsed line unless marked. Each "new" prediction is the theory's. Old-format values are what I expect the baseline to be; they are not part of the test.

| model | baseline, step 1 (k = 1) | predicted old, C3 | **predicted new, C3** | predicted new, C2 | other predictions |
|---|---|---|---|---|---|
| gemma4:12b | 0/100 invalid | 1–3% repeated-letter lines | **≤ 0.5%** | ≤ 0.5% | OLD correct on ≥ 90% of lines; C1 unchanged at about 0 |
| qwen2.5:7b | 0/100 invalid | 8–14% (calibration against GA 11.8%) | **≤ 2%** | ≤ 2% | OLD correct on ≥ 80% of lines (low confidence); the largest absolute drop of the four |
| mistral:7b | 42/138 attempts invalid; 4 repeated-letter | 2–8% | **≤ 1%** | ≤ 1% | wrong-line-count failures **not** reduced by more than half |
| llama3.2:3b | 207/266 attempts invalid (fallback 41/100); 1 repeated-letter | 2–8% of parsed lines | **≤ 1%** | ≤ 1% | **total invalid-attempt rate does NOT collapse**: it stays at ≥ 70% of its old value in C1–C3, because its failure is the wrong number of lines, which the OLD slot does not address. Fallbacks stay ≥ 25/100 in C1. OLD correct on < 80% of lines (low confidence) |

**For llama3.2:3b explicitly:** this test does not address its dominant failure. Its total invalid rate is predicted **not** to fall. Only its rare repeated-letter lines are predicted to fall. llama and mistral therefore serve as a specificity control. If the OLD slot also cut their wrong-line-count failures by more than half, the effect would be "more structure helps generally", not the specific mechanism tested here. That would be reported as such.

## What would confirm, and what would falsify

**Confirms the theory:**
- in C3, the repeated-letter rate falls by at least 75% from old to new in each model that has at least 10 repeated-letter lines under the old format;
- AND it falls in both C2 and C3 for qwen;
- AND llama's and mistral's wrong-line-count failures do not fall by more than half.

**Falsifies the theory:** any one of
- qwen's repeated-letter rate in C3 falls by less than 50%, or rises;
- gemma's falls by less than 50% in C3 while gemma states OLD correctly on at least 90% of lines;
- in two or more models, more than 1% of lines state OLD **correctly** and then give NEW = OLD. The format slot is filled correctly and the rule is still broken in the same line, which is exactly what "format binds where prose does not" says should not happen.

**Inconclusive, and reported as such:**
- old qwen C3 does not reproduce a repeated-letter rate of at least 5% (the probe does not elicit the GA failure);
- OLD accuracy below 50% in a model (the model cannot read the slot it is asked to fill, so the format change is not delivered).

## What the two outcomes mean

- **If the repeated-letter rate collapses in every model where it occurs:** that strongly supports the finding that format beats wording, as a property of these operators and not a quirk of one prompt. It holds across four model families, including gemma, which already partly obeyed the prose.
- **If it does not collapse:** the format-over-content finding has a problem, and this file says so in advance. The finding was built on cases where the invalid answer became impossible to express (`segment`, `position`). Here the invalid answer stays expressible but is made explicit. A failure here would mean that format binds only when it removes the option, not when it states it.
- **Either way:** llama3.2:3b's total invalid rate is predicted not to collapse. Its failure is a different one, and its result should not be read as evidence for or against this theory.

## Cost (planned)

- 4 models × 3 conditions × 2 formats: 1,000 calls per model, 4,000 calls in all.
- Requests including retries: an estimated 5,500–7,000.
- About 2.5–3.5 hours, one model loaded at a time.
- No ESMFold and no GA. Peak VRAM is one model: gemma4:12b at 8,171 MiB.
- All four models are already pulled.
