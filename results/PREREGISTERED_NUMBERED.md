# Pre-registered predictions: does a numbered sequence display fix the position-reading failure?

**This file is a prediction, not a result.** Drafted 2026-10-02. It will be committed before any call in the numbered display is made, and not edited afterwards. Outcomes will be scored in a separate file that cites it.

## The question

`results/INDEXING_NOT_INSTRUCTION.md` found that the repeated-letter failure is a failure to locate a position:
- no model repeats a letter it has read correctly (0 of 3,784 lines);
- every repeat is on a misread line;
- gemma4:12b reads positions 0–9 perfectly and positions 20+ at 0.23;
- the three smaller models read poorly almost everywhere at three changes per call.

That finding is observational. Correctly read lines are not a random sample.

**The test:** make locating easy, and see whether the failure goes away. The sequence is shown with every letter labelled by its index (`0:A 1:C 2:D …`) instead of space-separated (`A C D …`).
- If the failure is a locating failure, it should largely disappear, and reading accuracy should become flat across positions.
- If it does not disappear while reading accuracy does rise, then reading was not the cause.

## The change

One line of the base `mutate/position` prompt changes: the sequence line.

```
current:   Sequence (0-indexed positions 0-62): A C D E ...
numbered:  Sequence (0-indexed positions 0-62): 0:A 1:C 2:D 3:E ...
```

- Everything else is byte-identical: instructions, the MUST-differ sentence, the response format, the retry hint and the system prompt.
- It is built as `sequence_model_oldfield.py` was: from the base plan, with an assertion that the line occurs exactly once, and a verified inverse.
- The response parser is the base parser for **numbered/base**, and the OLD-slot parser for **numbered/OLD**.
- One unavoidable side effect: the prompt is about 3x longer in sequence tokens. It still fits `num_ctx` 4096 comfortably.

## Design

| cell | display | response format | source |
|---|---|---|---|
| spaced/base | spaced (production) | `POSITION, NEW` | the OLD-slot gate's old-format cells (`results/raw/oldfield_gate`), reused |
| spaced/OLD | spaced | `POSITION, OLD, NEW` | the gate's new-format cells, reused |
| **numbered/base** | numbered | `POSITION, NEW` | **new** |
| **numbered/OLD** | numbered | `POSITION, OLD, NEW` | **new** |

- **Conditions:** C2 (200 random length-63 sequences, k = 3) and C3 (the gate's 200 replayed GA sequences, k = 2–4). These are the gate's exact inputs and request-seed streams.
- **Models:** gemma4:12b, qwen2.5:7b, mistral:7b, llama3.2:3b.
- **Drift check:** a C1 rerun in the spaced/base format per model, 100 calls.
- **Calls:** 900 per model, 3,600 in all. Temperature 0.7, seed 0, retries 2, timeout 600 s.
- **Reusing the gate's spaced cells is justified** because the C1 replication reproduced prompts byte for byte across ten days for all four models. The drift check confirms this again before the reused cells are trusted (rule 1).

## Measures (all on FIRST ATTEMPTS only; all-attempts numbers are reported but not scored)

- **Repeated-letter rate:** parsed lines whose NEW equals the letter actually at that position.
- **Reading accuracy:** numbered/OLD lines whose OLD equals the actual letter, overall and by position band 0–9 / 10–19 / 20+.
- **Conditional repeat rate:** among numbered/OLD lines with OLD correct, the share with NEW equal to that letter.

## Predictions

| model | spaced/base repeated-letter rate, C3 (measured in the gate) | **numbered/base, C3** | spaced/OLD reading (gate) | **numbered/OLD reading** |
|---|---|---|---|---|
| gemma4:12b | 0.5% (3 lines) | ≤ 0.5% | 0.62 overall; 0.23 at 20+ | **≥ 0.95 in every band** |
| qwen2.5:7b | 4.1% (26 lines) | **≤ 1.0%** | 0.13 | **≥ 0.80** |
| mistral:7b | 5.4% (24 lines) | **≤ 1.5%** | 0.09 | ≥ 0.50 (low confidence) |
| llama3.2:3b | 5.0% (32 lines) | **≤ 1.5%** | 0.08 | ≥ 0.70 |

Further predictions:
- **C2** follows the same direction as C3.
- **The conditional repeat rate** in numbered/OLD stays ≤ 0.5% in every model.
- **gemma's band profile under numbered/OLD** is flat: the 20+ band is within 0.05 of the 0–9 band.

## Rules, and the order in which they apply

The order is fixed in advance. Each rule is applied only if every earlier rule allows the analysis to continue.

1. **Drift check.**
   - If a model's spaced/base C1 prompts do not reproduce the gate's C1 prompts byte for byte, that model's gate cells are not reused.
   - Its spaced cells are rerun in this session before any comparison, and the result is reported as such.
2. **Delivery, per model.**
   - A model is **delivered** if its numbered/OLD reading accuracy in C3 is ≥ 0.50.
   - A model that is not delivered is reported, and excluded from rules 3 and 4.
   - For that model the numbered display did not make locating reliable, so its repeat rate says nothing about whether reading is the cause.
3. **Falsification, checked before any confirmation, over delivered models only.** The claim "the failure is a reading failure" is **FALSIFIED** if any of:
   - (a) a delivered model with ≥ 10 repeated lines in spaced/base C3 has its numbered/base C3 repeated-letter rate fall by **less than 50%** (it reads, and still fails);
   - (b) in two or more delivered models, the conditional repeat rate in numbered/OLD exceeds **1%** (it reads the letter, and still repeats it).
   - **If either holds, the verdict is FALSIFIED, regardless of rule 4.**
4. **Confirmation, only if rule 3 found nothing.** The verdict is **CONFIRMED** if:
   - at least two delivered models have ≥ 10 repeated lines in spaced/base C3,
   - AND each such model's numbered/base C3 rate falls by ≥ 75%,
   - AND the same direction holds in C2.
5. **Otherwise the verdict is INCONCLUSIVE.** That includes fewer than two delivered models with enough baseline repeats, or drops between 50% and 75%. The reasons are stated.

Not rules (reported, not scored):
- the probe-to-GA calibration;
- token and latency costs;
- gemma's band profile;
- changes in which positions are chosen. The relabelling result says the labels shown steer the choice, so the numbered display may move the favourite positions.

## What each verdict means

- **CONFIRMED:** making positions easy to locate removes the repeated-letter failure in the models that could then locate. That establishes, by intervention rather than observation, that the failure is a reading failure.
- **FALSIFIED:** the models could locate and still failed. The observational pattern in `INDEXING_NOT_INSTRUCTION.md` would then have to be a selection effect, and that claim is wrong.
- **INCONCLUSIVE:** the display did not make locating reliable, or the change was too small to decide.
- **None of the three verdicts says anything about search quality.** This is an isolated probe, with no GA and no fitness.

## Cost (planned)

- 3,600 calls, one model at a time, peak VRAM 8,171 MiB (gemma), no ESMFold.
- The gate took about 11 hours for 4,000 calls, driven by retries in the OLD-slot cells. This should take 5–10 hours, depending on the numbered/OLD retry rate.
- The same refusal to load while another user's process is on the GPU. If that stops the run, it is resumed per model, as in the gate.
