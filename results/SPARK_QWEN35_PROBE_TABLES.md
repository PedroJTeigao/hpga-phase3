# Spark probe, nvidia/Qwen3.6-35B-A3B-NVFP4: generated tables (experiments/analyse_spark_qwen_probe.py)

Not a controlled model-scale comparison: a different model from the one requested, served by vLLM on a DGX Spark, not by Ollama. Source: results/raw/spark_qwen/. Read with results/SPARK_QWEN35_PROBE.md.

## 1. Log checks against the committed code

- Replay bases in the Spark gate directory identical to results/raw/oldfield_gate/oldfield_replay_bases.json: True

| cell | calls | attempts | fallbacks | transport errors | model field | temperature | attempt structure ok | `valid` flags not reproduced by the committed parser | prompts not rebuilt byte for byte by the committed code | inputs equal the fixed C2/C3 lists | request-seed stream replays |
|---|---|---|---|---|---|---|---|---|---|---|---|
| C1_old | 100 | 102 | 0 | 0 | nvidia/Qwen3.6-35B-A3B-NVFP4 | [0.7] | True | 0 | 0 | n/a (C1) | True |
| C1_new | 100 | 101 | 0 | 0 | nvidia/Qwen3.6-35B-A3B-NVFP4 | [0.7] | True | 0 | 0 | n/a (C1) | True |
| C2_old | 200 | 202 | 0 | 0 | nvidia/Qwen3.6-35B-A3B-NVFP4 | [0.7] | True | 0 | 0 | True | True |
| C2_new | 200 | 495 | 104 | 0 | nvidia/Qwen3.6-35B-A3B-NVFP4 | [0.7] | True | 0 | 0 | True | True |
| C3_old | 200 | 206 | 0 | 0 | nvidia/Qwen3.6-35B-A3B-NVFP4 | [0.7] | True | 0 | 0 | True | True |
| C3_new | 200 | 487 | 101 | 0 | nvidia/Qwen3.6-35B-A3B-NVFP4 | [0.7] | True | 0 | 0 | True | True |
| C1_drift | 100 | 101 | 0 | 0 | nvidia/Qwen3.6-35B-A3B-NVFP4 | [0.7] | True | 0 | 0 | n/a (C1) | True |
| C2_num | 200 | 200 | 0 | 0 | nvidia/Qwen3.6-35B-A3B-NVFP4 | [0.7] | True | 0 | 0 | True | True |
| C2_numold | 200 | 205 | 1 | 0 | nvidia/Qwen3.6-35B-A3B-NVFP4 | [0.7] | True | 0 | 0 | True | True |
| C3_num | 200 | 200 | 0 | 0 | nvidia/Qwen3.6-35B-A3B-NVFP4 | [0.7] | True | 0 | 0 | True | True |
| C3_numold | 200 | 205 | 1 | 0 | nvidia/Qwen3.6-35B-A3B-NVFP4 | [0.7] | True | 0 | 0 | True | True |

## 2. C1 drift check (spaced/base C1 of the Spark gate against the Spark numbered probe's C1_drift)

- First-attempt prompts: 100 of 100 identical (and 100 of 100 request seeds identical).
- All-attempt prompt streams: 102 against 101 prompts; identical as lists: False (the test `probe_mutate_numbered.py summarize` applies).
- Retries in C1_old at input_index [44, 62]; in C1_drift at [44].
- Same prompt and same request seed, different response: 17 of 100 first-attempt pairs.
- Spark C1_old first-attempt prompts against the 2026-10-01 gate's gemma4_12b C1_old first attempts: 100 of 100 identical.
- Spark C1_old first-attempt prompts against the 2026-10-01 gate's qwen2.5_7b C1_old first attempts: 100 of 100 identical.
- Spark C1_old first-attempt prompts against the 2026-10-01 gate's mistral_7b C1_old first attempts: 1 of 100 identical.
- Spark C1_old first-attempt prompts against the 2026-10-01 gate's llama3.2_3b C1_old first attempts: 1 of 100 identical.

## 3. First attempts (the pre-registered scope)

| cell | display/response | attempts | valid | lines | NEW = current letter (rate) | reading (OLD correct) | reading by band 0-9 / 10-19 / 20+ | repeats on correctly read lines | repeats on misread lines |
|---|---|---|---|---|---|---|---|---|---|
| C1_old | spaced/base | 100 | 98 | 99 | 1 (0.0101) |  |  |  |  |
| C1_new | spaced/OLD | 100 | 99 | 100 | 0 (0.0000) | 99/100 = 0.99 | 87/88 = 0.99 / 12/12 = 1.00 | 0 | 0 |
| C2_old | spaced/base | 200 | 199 | 600 | 1 (0.0017) |  |  |  |  |
| C2_new | spaced/OLD | 200 | 34 | 600 | 14 (0.0233) | 337/600 = 0.56 | 231/231 = 1.00 / 61/140 = 0.44 / 45/229 = 0.20 | 0 | 14 |
| C3_old | spaced/base | 200 | 195 | 640 | 5 (0.0078) |  |  |  |  |
| C3_new | spaced/OLD | 200 | 36 | 640 | 15 (0.0234) | 357/640 = 0.56 | 246/247 = 1.00 / 66/142 = 0.46 / 45/251 = 0.18 | 0 | 15 |
| C1_drift | spaced/base (drift) | 100 | 99 | 100 | 1 (0.0100) |  |  |  |  |
| C2_num | numbered/base | 200 | 200 | 600 | 0 (0.0000) |  |  |  |  |
| C2_numold | numbered/OLD | 200 | 196 | 600 | 0 (0.0000) | 596/600 = 0.99 | 230/230 = 1.00 / 147/149 = 0.99 / 219/221 = 0.99 | 0 | 0 |
| C3_num | numbered/base | 200 | 200 | 640 | 0 (0.0000) |  |  |  |  |
| C3_numold | numbered/OLD | 200 | 196 | 640 | 1 (0.0016) | 636/640 = 0.99 | 227/227 = 1.00 / 149/152 = 0.98 / 260/261 = 1.00 | 0 | 1 |

Correctly read lines, all OLD-slot cells (C1_new, C2_new, C3_new, C2_numold, C3_numold): 2025; of those with NEW = the letter just stated: 0. C2 and C3 only: 1926.

Invalid first attempts in the OLD-slot cells, by reason (the strict parser rejects a line whose OLD is not the letter at that position, hpga/sequence_model_oldfield.py:77):

- C1_new: {'OLD wrong, every NEW differs': 1}
- C2_new: {'OLD wrong, every NEW differs': 152, 'OLD wrong, and a NEW equals the current letter': 14}
- C3_new: {'OLD wrong, every NEW differs': 149, 'OLD wrong, and a NEW equals the current letter': 15}
- C2_numold: {'OLD wrong, every NEW differs': 4}
- C3_numold: {'OLD wrong, every NEW differs': 3, 'OLD wrong, and a NEW equals the current letter': 1}

## 4. All attempts (reported, not scored)

| cell | display/response | attempts | valid | lines | NEW = current letter (rate) | reading (OLD correct) | reading by band 0-9 / 10-19 / 20+ | repeats on correctly read lines | repeats on misread lines |
|---|---|---|---|---|---|---|---|---|---|
| C1_old | spaced/base | 102 | 100 | 101 | 1 (0.0099) |  |  |  |  |
| C1_new | spaced/OLD | 101 | 100 | 101 | 0 (0.0000) | 100/101 = 0.99 | 88/89 = 0.99 / 12/12 = 1.00 | 0 | 0 |
| C2_old | spaced/base | 202 | 200 | 606 | 2 (0.0033) |  |  |  |  |
| C2_new | spaced/OLD | 495 | 96 | 1485 | 24 (0.0162) | 872/1485 = 0.59 | 583/583 = 1.00 / 194/384 = 0.51 / 95/518 = 0.18 | 0 | 24 |
| C3_old | spaced/base | 206 | 200 | 661 | 6 (0.0091) |  |  |  |  |
| C3_new | spaced/OLD | 487 | 99 | 1577 | 35 (0.0222) | 933/1577 = 0.59 | 613/615 = 1.00 / 218/394 = 0.55 / 102/568 = 0.18 | 0 | 35 |
| C1_drift | spaced/base (drift) | 101 | 100 | 101 | 1 (0.0099) |  |  |  |  |
| C2_num | numbered/base | 200 | 200 | 600 | 0 (0.0000) |  |  |  |  |
| C2_numold | numbered/OLD | 205 | 199 | 615 | 1 (0.0016) | 609/615 = 0.99 | 240/240 = 1.00 / 148/152 = 0.97 / 221/223 = 0.99 | 0 | 1 |
| C3_num | numbered/base | 200 | 200 | 640 | 0 (0.0000) |  |  |  |  |
| C3_numold | numbered/OLD | 205 | 199 | 655 | 1 (0.0015) | 649/655 = 0.99 | 234/234 = 1.00 / 151/156 = 0.97 / 264/265 = 1.00 | 0 | 1 |

Correctly read lines, all OLD-slot cells (C1_new, C2_new, C3_new, C2_numold, C3_numold): 3163; of those with NEW = the letter just stated: 0. C2 and C3 only: 3063.

## 5. Execution (from the per-model JSON files; wall-clock time, nothing records GPU utilisation; no folding ran)

| probe | cell | calls | HTTP requests | fallbacks | wall s |
|---|---|---|---|---|---|
| gate | C1_old | 100 | 102 | 0 | 27.3 |
| gate | C1_new | 100 | 101 | 0 | 33.7 |
| gate | C2_old | 200 | 202 | 0 | 110.9 |
| gate | C2_new | 200 | 495 | 104 | 350.9 |
| gate | C3_old | 200 | 206 | 0 | 116.4 |
| gate | C3_new | 200 | 487 | 101 | 368.6 |
| numbered | C1_drift | 100 | 101 | 0 | 27.0 |
| numbered | C2_num | 200 | 200 | 0 | 114.0 |
| numbered | C2_numold | 200 | 205 | 1 | 154.8 |
| numbered | C3_num | 200 | 200 | 0 | 117.8 |
| numbered | C3_numold | 200 | 205 | 1 | 156.9 |
