# Both operator probes on `nvidia/Qwen3.6-35B-A3B-NVFP4` (DGX Spark, vLLM)

*Written 2026-10-13. The run was made by Italo on 2026-10-06 on a DGX Spark and handed over as a ZIP (`spark_qwen_probes_2026-10-06.zip`, md5 `2fa8780219489ee6063a15699d001b5a`). The raw output and his handoff note are in `results/raw/spark_qwen/`, unmodified. Every number below was re-derived here from those JSONL files by `experiments/analyse_spark_qwen_probe.py`, whose output is `results/SPARK_QWEN35_PROBE_TABLES.md`. Numbers computed earlier off this machine were used only as a cross-check, and all of them agree. Scope: first attempts, C3 unless stated, as in `PREREGISTERED_NUMBERED.md`.*

> **Which model this is.** It is `nvidia/Qwen3.6-35B-A3B-NVFP4`, **not** the 26B model that was requested. Identify it by that exact name. It is **not a controlled model-scale comparison** with anything else in this repository: it differs from every earlier model in size, architecture, quantisation, generation and serving stack all at once (see "What this run does not answer").
>
> **Not in the pre-registration.** `PREREGISTERED_NUMBERED.md` names four models. Its rules are applied to this fifth model below **by analogy only**. That application cannot alter the verdict scored for those four.

## Provenance and settings

- **Who and when.** Run by Italo on 2026-10-06. Gate: 13:50:02–14:06:51 (−0700). Numbered probe: 14:41:14–14:50:45. Both worker processes returned 0.
- **Endpoint.** `http://100.64.195.14:8000/v1`, an OpenAI-compatible vLLM server on the Spark (`backend: openai`, `remote: true`, `use_proxy: false`).
- **Model string.** `nvidia/Qwen3.6-35B-A3B-NVFP4`, in both metadata files and in the `model` field of every logged attempt.
- **No weight digest.** The remote API recorded none, so the exact weights cannot be pinned. The Ollama runs recorded a digest per model; this run cannot.
- **Settings, identical in `oldfield_meta.json` and `numbered_meta.json`:**
  - temperature 0.7;
  - `max_retries` 2, so up to three attempts per call;
  - `timeout_s` 600;
  - `context_tokens` 4096. Per the handoff note this is a legacy client setting that is not sent to vLLM; the server manages its own context. Nothing in the files can confirm or refute that.
  - `extra_body: {"chat_template_kwargs": {"enable_thinking": false}}`, the non-thinking Qwen template;
  - `PYTHONHASHSEED=0`. `HPGA_LLM_TIMEOUT_S`, `HPGA_LLM_TEMPERATURE` and `HPGA_LLM_NUM_CTX` are unset;
  - warm-up requests took 0.20 s each and are excluded from the logs.
- **The client is not in this repository.** No branch has an OpenAI backend: `hpga/operators.py` calls Ollama only. The metadata files also have a different shape from what the committed drivers write: no `ollama_version`, no digest, and extra `backend`/`endpoint`/`llm_settings` fields. The logs nevertheless match the committed code everywhere they can be checked (next section). Two things cannot be checked: the system prompt, which is not logged, and whether the per-request seed reached the server.
- **Cost.** No folding ran, so there is no fold time. Wall-clock only: 16 min 49 s for the gate's 1,000 calls (1,593 requests) and 9 min 31 s for the numbered probe's 900 calls (911 requests). Nothing in the files records GPU utilisation.

## Log checks (`SPARK_QWEN35_PROBE_TABLES.md` §1)

In all 11 cells:
- 0 transport errors;
- every attempt carries the model string above and temperature 0.7;
- attempts are numbered 0, 1, 2 within each call, with a fallback exactly when the third attempt is invalid.

Against the committed code:
- **Prompts.** All 2,504 logged prompts, retries included, are rebuilt byte for byte by `hpga/sequence_model.py`, `sequence_model_oldfield.py` and `sequence_model_numbered.py`.
- **Validity.** Every `valid` flag is reproduced by the committed parsers.
- **Inputs.** The C2 and C3 inputs equal the fixed lists. The replay-base file in the gate directory is identical to `results/raw/oldfield_gate/oldfield_replay_bases.json`.
- **Seeds.** The request-seed stream replays from `Random(0)`, fallbacks included.
- **Against the 2026-10-01 gate.** The Spark C1_old first-attempt prompts are identical to the 2026-10-01 gate's for gemma4:12b and qwen2.5:7b (100/100). They differ from mistral's and llama's from call 1 on, because those runs retried call 0 (llama then fell back on it) and the C1 stream is shared with the sequences.

## 1. What "valid" means in the OLD-slot cells

A first attempt in an OLD-slot cell is valid only if every stated OLD letter equals the letter actually at that position.
- `hpga/sequence_model_oldfield.py:77`: `if old != base[pos] or new == base[pos]: return None`. The module docstring states the same at line 21: "OLD equal to the letter actually at that position, NEW different from it".
- That `extract` is the plan's parser (line 92).
- `_run_llm_op` records `"valid": parsed is not None` (`hpga/operators.py:490`), retries on `None`, and after the third `None` falls back to the deterministic operator (lines 499–507).
- The numbered/OLD cells use the same parser: `sequence_model_numbered.py:47` builds them on the OLD-slot plan and keeps its `parse`.

**Consequence.** A misread line is rejected even when its NEW letter is a perfectly good change.

| cell | invalid first attempts | OLD wrong, every NEW differs | OLD wrong, and a NEW equals the current letter | any other reason |
|---|---|---|---|---|
| spaced/OLD C2 | 166 / 200 | 152 | 14 | 0 |
| spaced/OLD C3 | 164 / 200 | 149 | 15 | 0 |
| numbered/OLD C2 | 4 / 200 | 4 | 0 | 0 |
| numbered/OLD C3 | 4 / 200 | 3 | 1 | 0 |

- **Every invalid first attempt in these cells has a wrong OLD.** 149–152 of 200 would have passed the base format's rule.
- Retries mostly misread again. That is why 104 (C2) and 101 (C3) of 200 spaced/OLD calls ended in deterministic fallback.
- **This is not a failed run.** The handoff note's execution table, read without this, looks like one: 205 fallbacks and about 2.5 requests per call in the spaced/OLD cells. In fact the strict format rejected the model's misreadings, which is what the gate was built to expose.

## Results (first attempts)

**Reading accuracy** (stated OLD = the actual letter), by position band 0–9 / 10–19 / 20+:

| cell | overall | 0–9 | 10–19 | 20+ |
|---|---|---|---|---|
| spaced/OLD C3 | 357/640 = 0.56 | 246/247 = 1.00 | 66/142 = 0.46 | 45/251 = 0.18 |
| spaced/OLD C2 | 337/600 = 0.56 | 231/231 = 1.00 | 61/140 = 0.44 | 45/229 = 0.20 |
| numbered/OLD C3 | 636/640 = 0.99 | 227/227 = 1.00 | 149/152 = 0.98 | 260/261 = 1.00 |
| numbered/OLD C2 | 596/600 = 0.99 | 230/230 = 1.00 | 147/149 = 0.99 | 219/221 = 0.99 |
| *gemma4:12b spaced/OLD C3, for reference* | *0.62* | *1.00* | *0.73* | *0.23* |

(The gemma4:12b row is from `INDEXING_NOT_INSTRUCTION.md` Evidence 3. C1 spaced/OLD reads 99/100, but C1's chosen positions are all below 20.)

**Valid first attempts, fallbacks, and repeats** (lines whose NEW equals the letter actually at that position):

| cell | valid first attempts | fallbacks (calls) | repeated-letter lines |
|---|---|---|---|
| spaced/base C2 / C3 | 199/200 / 195/200 | 0 / 0 | 1/600 / **5/640** |
| spaced/OLD C2 / C3 | 34/200 / 36/200 | 104 / 101 | 14/600 / 15/640 |
| numbered/base C2 / C3 | 200/200 / 200/200 | 0 / 0 | 0/600 / 0/640 |
| numbered/OLD C2 / C3 | 196/200 / 196/200 | 1 / 1 | 0/600 / 1/640 |

**Conditional repeats.** Among lines whose OLD was stated correctly, 0 named that same letter as NEW. That is 0 of 2,025, summed over all five OLD-slot cells:

| cell | correctly read lines |
|---|---|
| spaced C1 | 99 |
| spaced C2 | 337 |
| spaced C3 | 357 |
| numbered C2 | 596 |
| numbered C3 | 636 |
| **total** | **2,025** |

C2 and C3 alone give 1,926. Over all attempts it is 0 of 3,163.

Every repeat in an OLD-slot cell is on a misread line. In the spaced cells that is 14 of 263 misread lines (5.3%) in C2 and 15 of 283 (5.3%) in C3. That is near 1/19, but no chance baseline was computed here.

## The pre-registered rules, applied by analogy

**Rule 1, drift. Not passed as implemented: a deviation, reported here.**

- **What the rule said.** `PREREGISTERED_NUMBERED.md` rule 1: "If a model's spaced/base C1 prompts do not reproduce the gate's C1 prompts byte for byte, that model's gate cells are not reused. Its spaced cells are rerun in this session before any comparison, and the result is reported as such."
  - The implementation compares the whole logged prompt list, retries included: `experiments/probe_mutate_numbered.py:252–254`, `"pass": a == b`.
  - The 2026-10-02 scoring applied it that way. Its counts (mistral 138/138, llama 266/266) include retries.
- **What happened.**
  - First-attempt prompts: 100 of 100 identical between the Spark gate's C1_old and the Spark numbered probe's C1_drift, and so are the 100 request seeds.
  - All-attempt streams: 102 against 101 prompts, so the implemented test returns `False`. C1_old retried calls 44 and 62; C1_drift retried call 44 only.
  - At call 62 the same prompt and the same seed produced `3: Q` (invalid) in the gate and a valid line in the drift cell.
  - In 17 of the 100 first-attempt pairs, the same prompt and seed gave a different response.
  - So on this endpoint the all-attempt stream depends on server-side variation, not only on prompt construction. Under Ollama the four models' all-attempt streams reproduced exactly. Whether the client sends the seed to vLLM cannot be checked, because the client is not in the repository.
- **The 100/100 first-attempt match is partly luck.**
  - C1 uses one random stream for both the sequences and the request seeds, so the gate's extra retry at call 62 drew one more number than the drift cell did.
  - The stream realigned only because `random.choice`'s rejection sampler discarded that draw: its top 5 bits are 30, which is ≥ 20.
  - Had the draw been accepted, calls 63–99 would have had different sequences in the two cells.
- **Effect on the numbers: none.**
  - Unlike the 2026-10-02 probe, nothing here reuses spaced cells from another session.
  - The Spark spaced cells were run on the same endpoint 35 minutes before the numbered cells.
- **A proposal, not a pre-registered rule, and not applied as one.**
  - Rule 1 exists to show that prompt construction has not drifted between the sessions whose cells are compared.
  - On a server that does not reproduce responses for a fixed seed, test that purpose directly:
    - first-attempt prompts and seeds, which match 100/100 here;
    - plus a byte-for-byte rebuild of every logged prompt from the committed code, which gives 0 mismatches in 2,504 here.
  - Read that way, rule 1 is met. Separately, the remedy rule 1 prescribes when drift fails ("rerun in this session") is close to what was done anyway.
  - This file reports the strict outcome as **not passed**. Adopting the proposal is the project lead's decision, as a dated amendment.
- **What would decide the strict rule either way.** Rerun C1_old and C1_drift against a server that returns the same response for the same prompt and seed:
  1. confirm that the client sends the seed and that vLLM honours it for this model;
  2. then make the comparison.
  - Identical all-attempt streams would mean the rule is met as written.
  - A first-attempt mismatch would be real drift.

**Rule 2, delivery.** numbered/OLD C3 reading is 0.99, at least 0.50, so the model is delivered.

**Rule 3(a). Not testable with this model.**
- The baseline (spaced/base C3) has **5** repeated lines, below the pre-registered threshold of 10. C2 has 1.
- So, exactly as with gemma4:12b (3/640), this model **cannot decide rule 3(a)**.
- The baseline repeats do fall: 5 → 0 in C3 and 1 → 0 in C2 under the numbered display. That is the direction gemma4:12b and qwen2.5:7b showed, but on counts the pre-registration set aside as too small to score.
- The reading numbers above are dramatic. They are not a test of the pre-registered claim and must not be read as a confirmation of it.

**Rule 3(b).** Conditional repeats in numbered/OLD are 0/596 (C2) and 0/636 (C3), not above 1%.

**Rule 4** needs at least 10 baseline repeats, so it does not apply.

**For this model alone, by analogy: INCONCLUSIVE** (rule 5). It has no baseline with enough repeats.

## What this run establishes

Observational, single run per cell, one seed, temperature 0.7. Each point is stated separately.

**No significance claim.** With one seed, the project's every-seed rule has a sign-test floor of 2/2^s = 1, so nothing here is a significant comparison. These are descriptive counts.

1. **A fifth model reproduces the positional reading collapse on the spaced display.**
   - The profile is gemma-like at the ends: 1.00 at positions 0–9 and 0.18 at 20+ (C3; C2 1.00 and 0.20), against gemma4:12b's 1.00 and 0.23.
   - It is lower than gemma in the middle band: 0.46 against 0.73.
   - Relation to the earlier models:
     - it is a different family from gemma4:12b, mistral:7b and llama3.2:3b;
     - against qwen2.5:7b it is the same Qwen family, but a later generation and a mixture-of-experts architecture.
   - `INDEXING_NOT_INSTRUCTION.md`'s limit 4 applies: reading was measured with the OLD slot present.
2. **The numbered display raises this model's reading at positions 20+ to 1.00, and the profile goes flat.**
   - C3: 0.18 → 260/261 at 20+; 1.00 / 0.98 / 1.00 across the bands.
   - C2: 0.20 → 0.99 at 20+; 1.00 / 0.99 / 0.99 across the bands.
   - No prediction was registered for this model; the flatness criterion (20+ within 0.05 of 0–9) was registered for gemma4:12b only.
3. **The conditional result holds again.** 0 repeats among 2,025 correctly read lines (composition above). On first attempts, the pre-registered scope, all three experiments that measured this give **0 of 7,844 over five models** (gemma4:12b, qwen2.5:7b, mistral:7b, llama3.2:3b, and this model) **and two displays**:
   - **1,379** from the OLD-slot gate: four models, spaced display, C1–C3. This is a first-attempt recount; the published 3,784 is the all-attempts figure for the same gate;
   - **4,440** from the numbered probe: the same four models, numbered display, C2–C3 (`PREREGISTERED_NUMBERED_SCORED.md` rule 3(b));
   - **2,025** from this run: one model, spaced C1–C3 and numbered C2–C3.
   - All three use the same line definition, and no line is counted twice. Condition coverage differs by design: the numbered probe has no C1 OLD-slot cell. That makes the total a count of lines checked, not a rate.
   - On all attempts the same three give 0 of 3,784 + 6,718 + 3,163 = 13,665.
   - Both totals come from `experiments/recount_conditional_repeats.py`; the table is `results/CONDITIONAL_REPEATS_RECOUNT_TABLES.md`.
   - *Dated note, 2026-10-13: the first version of this file (`cb19567`) gave "0 of 10,249", summing 3,784 (all attempts) with 4,440 and 2,025 (first attempts). That mixed scopes and is replaced by the two totals above.*
   - It stays observational: correctly read lines are not a random sample (`INDEXING_NOT_INSTRUCTION.md` limit 5).

## What this run does not answer: whether the positional failure is a scale effect

- **Smaller per token, not larger.** "A3B" means about 3B active parameters per token (from the naming convention; no file here records a parameter count). In active compute per token this model is therefore *smaller* than gemma4:12b (11.9B, `OPERATOR_BEHAVIOUR.md`), although it is 35B in total.
- **Quantisation is not held fixed.** NVFP4 is a 4-bit format. The Ollama models were also 4-bit (Q4_K_M), but in a different quantisation scheme.
- **Too many differences at once.**
  - It is a different family from gemma4:12b and a much newer generation than any of the four models.
  - It runs on a different serving stack (vLLM, chat template with thinking disabled) and different hardware.
  - With all of these changing at once, a similarity or a difference to gemma4:12b cannot be attributed to size.
- **What would answer it.**
  - A **dense** model larger than 12B, in the **same family** as one already measured where possible: for example a larger dense qwen2.5 than 7b, or a larger dense gemma4 than 12b if one exists.
  - The same quantisation scheme and the same serving stack (Ollama) as its smaller sibling.
  - Run through the same two probes and compared on the reading-by-band profile.
  - For rule 3(a) to bite as well, the model would also need at least 10 baseline repeats in spaced/base C3.

## An engineering number: what the OLD slot costs as a safety check

If the OLD slot were used in a real run as a guard, meaning a mutate is accepted only if the model states the current letter correctly, then on this model:
- it would send **104/200 (52%) and 101/200 (50.5%)** of mutate calls to the deterministic fallback on the spaced display;
- it would send **1/200 (0.5%)** on the numbered display.
- The spaced/OLD cells also took about 2.5 requests per call and about 3.2x the wall time of spaced/base: 351–369 s against 111–116 s per 200 calls.

This is a property of the OLD-slot probe format, not of GA arm C. Arm C's production format (spaced/base) was **97.5–99.5% valid** on first attempts here (195/200 in C3, 199/200 in C2), with no fallbacks. On this model the slot is usable as a guard only together with the numbered display.

## Does this change the FALSIFIED verdict of `b87bee1`?

No.
- That verdict rests on rule 3(a) being met by mistral:7b and llama3.2:3b, and rule 3 makes FALSIFIED final "regardless of rule 4".
- This model does not qualify under 3(a): 5 baseline repeats, against a threshold of 10.
- It does not meet 3(b): 0% conditional repeats.
- It is outside the pre-registered model set in any case.
- Nothing it could have shown would reverse a falsification already met by two other models. A new pre-registration would be needed to ask a new question.

## Limits

- **One run per cell, one seed, temperature 0.7, one prompt family**, 100–200 calls per cell. No GA and no fitness: nothing here bears on search quality.
- **Not reproducible as run.**
  - The client is not in this repository.
  - The endpoint returned different responses for the same prompt and seed in 17 of 100 C1 pairs.
  - The model weights have no recorded digest.
  - The logs are checked against the committed code, but the run itself cannot be repeated from this repository.
- **The system prompt is not logged.** That it matches the committed one is assumed, not shown.
- **Not a controlled scale comparison** (above). The model was not the one requested.
- The probe-to-GA calibration and the other limits of `INDEXING_NOT_INSTRUCTION.md` and `PREREGISTERED_NUMBERED_SCORED.md` apply unchanged.
