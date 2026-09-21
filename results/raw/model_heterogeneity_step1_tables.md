### 1. Models, load time and GPU memory (num_ctx 4096, as the operators run)

| model | family | params | quant | disk (GB) | load time (s) | Ollama resident (GiB) | fully on GPU | nvidia-smi delta (MiB) |
|---|---|---|---|---|---|---|---|---|
| gemma4:12b | gemma4 | 11.9B | Q4_K_M | 7.56 | 10.7 | 7.51 | True | 8171 |
| llama3.2:3b | llama | 3.2B | Q4_K_M | 2.02 | 9.9 | 2.38 | True | 2563 |
| qwen2.5:7b | qwen2 | 7.6B | Q4_K_M | 4.68 | 14.7 | 4.42 | True | 4655 |
| mistral:7b | llama | 7.2B | Q4_K_M | 4.37 | 13.3 | 4.61 | True | 4847 |

Load time is Ollama's `load_duration` on a warm-up request after unloading everything (disk cache may be warm for some models).

### 2. Do two fit on the 15360 MiB T4 together? (each loaded by a tiny request, then the first re-touched)

| pair | both resident | both fully on GPU | combined nvidia-smi delta (MiB) | first model evicted or reloaded |
|---|---|---|---|---|
| gemma4:12b + llama3.2:3b | True | True | 10731 | False (retouch load 0.00 s) |
| gemma4:12b + qwen2.5:7b | True | True | 12823 | False (retouch load 0.00 s) |
| gemma4:12b + mistral:7b | True | True | 13015 | False (retouch load 0.00 s) |
| llama3.2:3b + qwen2.5:7b | True | True | 7215 | False (retouch load 0.00 s) |
| llama3.2:3b + mistral:7b | True | True | 7407 | False (retouch load 0.00 s) |
| qwen2.5:7b + mistral:7b | True | True | 9499 | False (retouch load 0.00 s) |

### 3. Format compliance (100 calls per operator, genome length 63, bounds [30, 80], temperature 0.7, seed 0)

| model | operator | fell back | fallback rate | requests / call | mean latency per request (s) | mean latency per call (s) | mean tokens out / request |
|---|---|---|---|---|---|---|---|
| gemma4:12b | mutate/position | 0/100 | 0.000 | 1.00 | 1.22 | 1.22 | 9.8 |
| gemma4:12b | crossover/segment | 0/100 | 0.000 | 1.00 | 3.72 | 3.72 | 21.0 |
| llama3.2:3b | mutate/position | 41/100 | 0.410 | 2.66 | 0.84 | 2.22 | 20.3 |
| llama3.2:3b | crossover/segment | 2/100 | 0.020 | 1.63 | 1.50 | 2.44 | 31.6 |
| qwen2.5:7b | mutate/position | 0/100 | 0.000 | 1.00 | 0.98 | 0.98 | 9.6 |
| qwen2.5:7b | crossover/segment | 0/100 | 0.000 | 1.00 | 2.73 | 2.73 | 21.0 |
| mistral:7b | mutate/position | 4/100 | 0.040 | 1.38 | 1.52 | 2.09 | 15.0 |
| mistral:7b | crossover/segment | 13/100 | 0.130 | 1.40 | 4.26 | 5.96 | 36.5 |

Fallback = the operator gave up after its retries and used the deterministic operator.

### 4. mutate/position: which positions and letters? (LLM-successful calls only)

| model | successful calls | distinct positions (of 63; expected if uniform) | modal position (share) | top-3 share | position chi2 (df 62), MC p | distinct new letters (of 20) | modal letter (share) | zero-change outputs |
|---|---|---|---|---|---|---|---|---|
| gemma4:12b | 100 | 17 (50.3) | 10 (38.0%) | 62.0% | 1083, p=5e-05 | 15 | G (29.0%) | 0 |
| llama3.2:3b | 59 | 21 (38.5) | 32 (23.7%) | 40.7% | 294, p=5e-05 | 15 | K (18.6%) | 13 |
| qwen2.5:7b | 100 | 6 (50.3) | 10 (55.0%) | 78.0% | 2102, p=5e-05 | 14 | D (25.0%) | 0 |
| mistral:7b | 96 | 40 (49.4) | 48 (6.2%) | 15.6% | 101, p=0.00155 | 11 | K (37.5%) | 3 |

Pooled over the 4 models: 355 successful mutations, 51 of 63 positions used by at least one model, modal position 10 (26.8%); positions that are the modal choice of each model: gemma4:12b: 10, llama3.2:3b: 32, qwen2.5:7b: 10, mistral:7b: 48.

### 5. crossover/segment: which cuts? (cuts are percentages; 99 possible values 1-99)

| model | valid declarations | segments per call | distinct cut values (of 99) | modal cut (share of cuts) | cuts within 45-55 | five most used cuts |
|---|---|---|---|---|---|---|
| gemma4:12b | 100 | {'2': 100} | 2 | 40 (82.0%) | 18.0% | 40x82, 50x18 |
| llama3.2:3b | 98 | {'2': 7, '3': 73, '4': 16, '5': 2} | 7 | 40 (46.9%) | 1.4% | 40x98, 80x85, 60x18, 20x3, 50x3 |
| qwen2.5:7b | 100 | {'2': 100} | 1 | 40 (100.0%) | 0.0% | 40x100 |
| mistral:7b | 87 | {'3': 87} | 5 | 40 (49.4%) | 0.0% | 40x86, 80x83, 85x3, 35x1, 88x1 |

Pooled over the 4 models: 583 cuts, 10 of 99 values used by at least one model (40x366, 80x168, 50x21, 60x18, 20x3, 85x3, 70x1, 90x1, 35x1, 88x1).

### 6. Do different models choose differently from each other?

**mutate/position** (choice = position; 'same (pos,letter)' also requires the same new letter)

| pair | positions used by both (of the union) | TV distance | permutation p | shared genomes | same position, observed | same position, shuffled baseline | same (pos,letter), observed |
|---|---|---|---|---|---|---|---|
| gemma4:12b vs llama3.2:3b | 9 of 29 | 0.828 | 0.0002 | 0 | n/a | n/a | n/a |
| gemma4:12b vs qwen2.5:7b | 6 of 17 | 0.440 | 0.0002 | 100 | 33.0% | 23.0% | 1.0% |
| gemma4:12b vs mistral:7b | 9 of 48 | 0.815 | 0.0002 | 1 | 0.0% | 0.0% | 0.0% |
| llama3.2:3b vs qwen2.5:7b | 3 of 24 | 0.898 | 0.0002 | 0 | n/a | n/a | n/a |
| llama3.2:3b vs mistral:7b | 15 of 46 | 0.661 | 0.0002 | 1 | 0.0% | 0.0% | 0.0% |
| qwen2.5:7b vs mistral:7b | 3 of 43 | 0.949 | 0.0002 | 1 | 0.0% | 0.0% | 0.0% |

**crossover/segment** (choice = the set of cut values in the declaration)

| pair | cut values used by both (of the union) | TV distance | permutation p | shared parent pairs | identical cut set, observed | identical cut set, shuffled baseline |
|---|---|---|---|---|---|---|
| gemma4:12b vs llama3.2:3b | 2 of 7 | 0.517 | 0.0002 | 10 | 0.0% | 0.0% |
| gemma4:12b vs qwen2.5:7b | 1 of 2 | 0.180 | 0.0002 | 100 | 82.0% | 82.0% |
| gemma4:12b vs mistral:7b | 1 of 6 | 0.506 | 0.0002 | 3 | 0.0% | 0.0% |
| llama3.2:3b vs qwen2.5:7b | 1 of 7 | 0.531 | 0.0002 | 10 | 0.0% | 0.0% |
| llama3.2:3b vs mistral:7b | 2 of 10 | 0.124 | 0.0566 | 1 | 100.0% | 100.0% |
| qwen2.5:7b vs mistral:7b | 1 of 5 | 0.506 | 0.0002 | 3 | 0.0% | 0.0% |

Permutation p-values use 5000 label shuffles (floor 2.0e-04); TV = total-variation distance (0 = identical distributions, 1 = disjoint). The shuffled baseline shuffles the second model's answers across the shared prompts (2000 shuffles): it is the agreement expected if the models share a marginal distribution but ignore the prompt.

### 7. Consistency check (log parse vs the probe's own JSON)

- gemma4:12b: 100 mutate calls and 100 crossover calls found in the log; probe counted 100 successful mutates; log-parse positions distinct 17 vs probe 17.
- llama3.2:3b: 100 mutate calls and 100 crossover calls found in the log; probe counted 59 successful mutates; log-parse positions distinct 21 vs probe 21.
- qwen2.5:7b: 100 mutate calls and 100 crossover calls found in the log; probe counted 100 successful mutates; log-parse positions distinct 6 vs probe 6.
- mistral:7b: 100 mutate calls and 100 crossover calls found in the log; probe counted 96 successful mutates; log-parse positions distinct 40 vs probe 40.
