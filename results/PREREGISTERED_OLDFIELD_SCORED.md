# Scoring the OLD-slot gate against its pre-registration

*This scores `results/PREREGISTERED_OLDFIELD.md`, committed as `bab55af` on 2026-10-01 before any call in the new format. That file is not edited. Code `15b115f`, committed before the first real call. Runs 2026-10-01 10:36–21:28. Raw data, the generated tables (`oldfield_tables.md`, `oldfield_summary.json`) and the logs are in `results/raw/oldfield_gate/`.*

## Run notes

- **Four launches.** The driver refuses to load a model while another user's process is on the GPU. That happened three times, between gemma and qwen, between qwen and mistral, and between mistral and llama. Each time the run stopped cleanly before the next model, and was resumed for the remaining models once the GPU was free.
  - **Correction, added 2026-10-02 (the sentence above is left as originally written, 2026-10-01):** the three stops were very likely **not** another user's process. They were very likely our own Ollama runner, still shutting down after the previous model was unloaded.
    - The driver checked the GPU immediately after unloading a model. It recognised our runner by `nvidia-smi`'s process-name field, which reads `[No data]` for an exiting runner. All three stops recorded exactly that: an unnamed `[No data]` process (`run_part*.log`).
    - The same false alarm stopped the numbered-display probe on 2026-10-02 at 15:34. The process it flagged (PID 3812868) had been listed 2 hours earlier as our own `/scratch/pcanaste/ollama/lib/ollama/llama-server`.
    - The behaviour was then reproduced live through three load/unload cycles: during unload, `nvidia-smi` shows our runner as `[No data]` and its executable link is unreadable.
    - The check was fixed in `experiments/probe_mutate_numbered.py` (commit `af74fb9`). The gate's own driver, `probe_mutate_oldfield.py`, still has the old check; it is not being run again.
    - **No measurement was affected.** Every stop fell between models, after one model's last cell had finished and before the next model's first call. No cell was interrupted, repeated or split across launches. The only cost was the delay before each resume.
- **No cell was interrupted or repeated.** Each model ran all six of its cells in one process.
- **Logs and metadata** for each part are kept: `run_part1_gemma.log` … `run_part4.log`, and `oldfield_meta_part*.json` plus the final `oldfield_meta.json`.
- **Model builds:** gemma `4eb23ef187e2…`, qwen `845dbda0ea48…`, mistral `6577803aa9a0…`, llama `a80c4f17acd5…`. Ollama 0.34.0.
- **Replication passed for all four.** The old-format C1 prompts match `MODEL_HETEROGENEITY_STEP1`'s logged mutate prompts byte for byte: 100/100, 100/100, 138/138 and 266/266, retries included. The old format here is the production path, and the models' behaviour reproduced exactly ten days later.

## The measured outcome (first attempts; all-attempts in the generated tables)

Repeated-letter rate = parsed lines whose NEW equals the letter actually at that position, scored the same way under both formats.

| model | C3 old | C3 new | C2 old | C2 new | OLD accuracy, new (C1 / C2 / C3) | strict invalid attempts, new (C2 / C3) |
|---|---|---|---|---|---|---|
| gemma4:12b | 3/640 (0.5%) | 8/640 (1.3%) | 5/600 (0.8%) | 5/600 (0.8%) | 0.72 / 0.62 / 0.58 | 175/200, 186/200 |
| qwen2.5:7b | 26/640 (4.1%) | 35/640 (5.5%) | 14/600 (2.3%) | 34/600 (5.7%) | 0.53 / 0.16 / 0.13 | 199/200, 200/200 |
| mistral:7b | 24/441 (5.4%) | 29/640 (4.5%) | 17/400 (4.3%) | 18/600 (3.0%) | 0.09 / 0.08 / 0.08 | 199/200, 200/200 |
| llama3.2:3b | 32/640 (5.0%) | 27/640 (4.2%) | 24/600 (4.0%) | 26/600 (4.3%) | 0.26 / 0.09 / 0.08 | 200/200, 200/200 |

**No model's repeated-letter rate collapsed.**
- It rose for gemma in C3 and for qwen in both C2 and C3.
- It fell by 17% (mistral) and 16% (llama) in C3.
- Strict validity collapsed for every model, because the OLD letter was usually wrong.

## Scores, prediction by prediction

| prediction | predicted | measured | score |
|---|---|---|---|
| gemma, old C3 | 1–3% | 0.5% | FALSE (below the range) |
| gemma, new C3 / C2 | ≤ 0.5% | 1.3% / 0.8% | **FALSE** |
| gemma, OLD accuracy | ≥ 90% | 0.58–0.72 | FALSE |
| gemma, C1 unchanged at about 0 | about 0 | 0/100 lines repeated; but 28/100 invalid attempts (wrong OLD) | TRUE for repeats; strict validity fell |
| qwen, old C3 (calibration against GA 11.8%) | 8–14% | 4.1% first attempts, 5.3% all attempts | FALSE (below the range) |
| qwen, new C3 / C2 | ≤ 2% | 5.5% / 5.7% | **FALSE** |
| qwen, OLD accuracy | ≥ 80% | 0.13–0.53 | FALSE |
| qwen, the largest absolute drop | yes | it rose | FALSE |
| mistral, old C3 | 2–8% | 5.4% | TRUE |
| mistral, new C3 / C2 | ≤ 1% | 4.5% / 3.0% | **FALSE** |
| mistral, wrong-line-count failures not reduced by more than half | not reduced by more than half | C2: 600 → 0; C3: 597 → 0 (old answers had about 2 lines for k = 3); C1: 38 → 109 | **FALSE in C2 and C3**, TRUE in C1 |
| llama, old C3 | 2–8% | 5.0% | TRUE |
| llama, new C3 / C2 | ≤ 1% | 4.2% / 4.3% | **FALSE** |
| llama, total invalid rate does not collapse (≥ 70% of old) | does not collapse | rose in every condition: C1 100 → 96 of 100 first attempts invalid; C2 40 → 200; C3 47 → 200 | **TRUE** |
| llama, C1 fallbacks ≥ 25/100 | ≥ 25 | 86 | TRUE |
| llama, OLD accuracy < 80% | < 80% | 0.08–0.26 | TRUE |

**The three outcome rules, applied as written:**
- **Confirms** (a ≥ 75% drop in every model with ≥ 10 repeated lines in old C3, which are qwen, mistral and llama; plus qwen falling in C2 and C3; plus llama's and mistral's line-count errors not falling by half): **not met.** The drops were −34% (a rise), 17% and 16%.
- **Falsifies**, any of:
  - qwen's C3 rate falls by less than 50% or rises: **met** (it rose);
  - gemma's C3 rate falls by less than 50% while gemma's OLD accuracy is ≥ 90%: not met (OLD accuracy 0.58);
  - in two or more models, more than 1% of lines with OLD correct also have NEW = OLD: **not met: 0 such lines in any model.**
- **Inconclusive**, any of:
  - OLD accuracy below 50% in a model: **met for qwen in C2 and C3, and for mistral and llama in all three conditions**;
  - old qwen C3 below 5%: met on first attempts (4.1%), not on all attempts (5.3%). The pre-registration did not say which.

So, by its own rules, the test both meets one falsification condition and is **inconclusive for three of the four models**: the format change was not delivered, because those models could not fill the slot. The file did not say which rule takes precedence, and I am not choosing one after seeing the data.

## What the gate actually shows

The pre-registered metric missed a pattern that runs through every model and is cleaner than anything the rules anticipated:

| model | lines with OLD stated correctly | of those, NEW = the current letter | lines with OLD wrong | of those, NEW = the actual letter |
|---|---|---|---|---|
| gemma4:12b | 2,176 | **0** | 1,281 | 40 (3.1%) |
| qwen2.5:7b | 704 | **0** | 3,292 | 205 (6.2%) |
| mistral:7b | 339 | **0** | 3,765 | 176 (4.7%) |
| llama3.2:3b | 565 | **0** | 3,950 | 155 (3.9%) |
| all | **3,784** | **0** | 12,288 | 576 (4.7%) |

*All three conditions, all attempts, new format. "Correct" means OLD equals the letter actually at the stated position.*

1. **When a model locates the letter at a position, it never repeats it.** That holds for 0 of 3,784 lines across four model families, three conditions, and the GA's own sequences. The prose rule "the new letter MUST differ" is obeyed without exception whenever the model knows what the letter is.
2. **Every repeated letter, all 576 of them, came on a line where the model had misread the letter at that position.** Those lines repeat the actual letter at 4.7%, about the chance rate (`BLIND_CHOICE.md` measures chance at roughly 5–10% in these data sets). So the repeated-letter failure is **a failure to locate the position, not a failure to follow the rule.**
3. **Locating fails often, and gets worse further along the sequence.**
   - gemma read the letter correctly at positions 0–9 almost always: 1 error in 196 lines in C3. It was wrong in most lines beyond position 20 (C3: 46/57 at 20–29, 78/88 at 40–49).
   - Its wrong OLD letters are most often the letter one position to the right: a counting error.
   - qwen, mistral and llama read the letter correctly in only 8–16% of lines at three changes per call, about what guessing gives for mistral and llama.
4. **This explains earlier results.**
   - gemma avoids the current letter at about 0.2x chance in the GA (`BLIND_CHOICE.md`) and favours low positions (10, 3, 0–3; `OPERATOR_BEHAVIOUR.md`). Its favourite positions are where it can still count.
   - qwen in the GA is near chance (`BLIND_CHOICE.md`) and in this gate mostly cannot locate letters at all at k = 3.
   - The difference between models in the GA's invalid rates (gemma 5.2%, qwen 32.8%) is plausibly a difference in **how well they count positions in a space-separated sequence**. That reading is consistent with these data; it was not tested.

**For the format-over-content finding** (`OPERATOR_BEHAVIOUR.md`). This gate does not support "format fixes what prose cannot" for this failure. It does not contradict it either, because the failure turns out not to be ignored prose. The models follow the prose rule perfectly when they can apply it. What they cannot reliably do is read the sequence at a stated index. The OLD slot made that visible, and it made strict validity collapse.

The project's central question, whether the operator reads the genome, gets a sharper answer here:
- at a position the model has located, yes, and it follows the rule about it;
- locating a position past the first 10–20 letters of a space-separated sequence fails most of the time, and for three of the four models it fails at every position when three changes are requested.

**For llama3.2:3b and mistral**, the predicted specificity controls:
- llama behaved as predicted: total invalid rate up, not down.
- mistral did not. The OLD slot removed its wrong-line-count failure entirely at k = 3: with the slot it gave the requested number of lines, without it about 2 lines. Its OLD letters were then wrong 92% of the time.
- So a more structured line changed how many lines one model wrote, a general structural effect, while fixing nothing about the letter.

## Limits

- **Isolated probe; no GA.** One prompt family, temperature 0.7, seed 0. 100–200 calls per cell, single run per cell.
- **The calibration was not met.** Old-format qwen on the replayed GA sequences repeated the letter in 4.1% of lines (5.3% counting retries), against 11.8% in its GA runs. The probe elicits the failure, but at half the GA rate or less; the GA rate depends on more than the sequences alone.
- **The conditional result (0 of 3,784) is observational.** Lines with a correct OLD are not a random sample: they are more often at early positions and from the stronger models. That they never repeat is a fact about these lines; it does not show what the same model would do at a position it could not locate if it were told the letter.
- **That test is the obvious next one, and it was not run:** show the letter (`POSITION 37 currently holds K`), or number the sequence (`37:K`). Candidate (c) of the design discussion.
- **Not tested:** whether gemma's position-counting errors grow with sequence length or with position; why mistral writes fewer lines than requested; whether a numbered display changes any of this.
