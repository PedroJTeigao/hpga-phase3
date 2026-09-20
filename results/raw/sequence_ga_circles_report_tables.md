### 9.1 Runs and the common budget

| seed | arm | distinct folds | cache hits | genomes evaluated (incl. repeats) | final best TM (own budget) | attempt |
|---|---|---|---|---|---|---|
| 0 | D GA + LLM + circles | 349 | 47 | 396 | 0.3639 | 1 |
| 0 | E GA + LLM + random immigrants | 358 | 38 | 396 | 0.4599 | 1 |
| 1 | D GA + LLM + circles | 342 | 54 | 396 | 0.6372 | 1 |
| 1 | E GA + LLM + random immigrants | 358 | 38 | 396 | 0.4848 | 1 |
| 2 | D GA + LLM + circles | 355 | 41 | 396 | 0.4726 | 1 |
| 2 | E GA + LLM + random immigrants | 358 | 38 | 396 | 0.4787 | 1 |
| 3 | D GA + LLM + circles | 356 | 40 | 396 | 0.5807 | 1 |
| 3 | E GA + LLM + random immigrants | 356 | 40 | 396 | 0.6146 | 1 |
| 4 | D GA + LLM + circles | 351 | 45 | 396 | 0.4034 | 1 |
| 4 | E GA + LLM + random immigrants | 358 | 38 | 396 | 0.4486 | 1 |

Distinct folds reached by each arm, and the common cut n_cut (the smallest of the five):

| seed | A | B | C | D | E | n_cut |
|---|---|---|---|---|---|---|
| 0 | 282 | 278 | 282 | 349 | 358 | 278 |
| 1 | 280 | 276 | 280 | 342 | 358 | 276 |
| 2 | 281 | 280 | 281 | 355 | 358 | 280 |
| 3 | 280 | 280 | 280 | 356 | 356 | 280 |
| 4 | 280 | 275 | 280 | 351 | 358 | 275 |

### 9.2 Best TM-score at the common distinct-fold count (per seed), and over seeds

| arm (read at n_cut) | seed 0 (n=278) | seed 1 (n=276) | seed 2 (n=280) | seed 3 (n=280) | seed 4 (n=275) |
|---|---|---|---|---|---|
| A random search | 0.3859 | 0.4004 | 0.4104 | 0.4035 | 0.4009 |
| B GA (deterministic ops) | 0.4203 | 0.5025 | 0.5040 | 0.4007 | 0.4873 |
| C GA (LLM ops) | 0.4632 | 0.4745 | 0.5138 | 0.5434 | 0.4370 |
| D GA + LLM + circles | 0.3587 | 0.6372 | 0.4726 | 0.5807 | 0.4015 |
| E GA + LLM + random immigrants | 0.4164 | 0.4719 | 0.4773 | 0.6099 | 0.4486 |

Each run's own final best over its full budget (D and E ran further than the cut; A, B, C as before):

| arm (full run) | seed 0 | seed 1 | seed 2 | seed 3 | seed 4 |
|---|---|---|---|---|---|
| A random search | 0.3859 | 0.4004 | 0.4104 | 0.4035 | 0.4009 |
| B GA (deterministic ops) | 0.4203 | 0.5025 | 0.5040 | 0.4007 | 0.4873 |
| C GA (LLM ops) | 0.4632 | 0.4745 | 0.5138 | 0.5434 | 0.4370 |
| D GA + LLM + circles | 0.3639 | 0.6372 | 0.4726 | 0.5807 | 0.4034 |
| E GA + LLM + random immigrants | 0.4599 | 0.4848 | 0.4787 | 0.6146 | 0.4486 |

Cross-seed summary at the common cut (the only pooled table; n = complete seeds):

| arm | n seeds | mean best-so-far at n_cut | range (min - max) |
|---|---|---|---|
| A random search | 5 | 0.4002 | 0.3859 - 0.4104 |
| B GA (deterministic ops) | 5 | 0.4630 | 0.4007 - 0.5040 |
| C GA (LLM ops) | 5 | 0.4864 | 0.4370 - 0.5434 |
| D GA + LLM + circles | 5 | 0.4901 | 0.3587 - 0.6372 |
| E GA + LLM + random immigrants | 5 | 0.4848 | 0.4164 - 0.6099 |

### 9.3 Mean pairwise edit distance within the evaluated population, per generation (per seed)

The evaluated population is 16 genomes at generation 0 and 20 afterwards for D and E (their 4 injected genomes included), 16 for B and C.

**Seed 0**

| gen | B | C | D | E |
|---|---|---|---|---|
| 0 | 50.3 | 50.3 | 50.3 | 50.3 |
| 1 | 42.0 | 42.4 | 45.7 | 45.8 |
| 2 | 40.9 | 38.9 | 42.9 | 42.7 |
| 3 | 38.2 | 26.4 | 44.1 | 40.2 |
| 4 | 31.1 | 27.5 | 46.6 | 43.1 |
| 5 | 18.6 | 28.5 | 47.3 | 38.2 |
| 6 | 14.3 | 28.8 | 43.2 | 36.0 |
| 7 | 14.8 | 21.1 | 41.2 | 35.9 |
| 8 | 15.5 | 18.3 | 41.0 | 40.1 |
| 9 | 14.2 | 17.1 | 40.2 | 38.4 |
| 10 | 15.8 | 10.0 | 34.2 | 39.6 |
| 11 | 14.1 | 9.2 | 37.2 | 36.5 |
| 12 | 16.8 | 8.0 | 32.3 | 35.5 |
| 13 | 14.4 | 7.5 | 38.6 | 32.6 |
| 14 | 18.4 | 7.5 | 36.7 | 31.0 |
| 15 | 18.4 | 9.0 | 34.5 | 34.9 |
| 16 | 17.6 | 8.1 | 33.4 | 38.1 |
| 17 | 14.6 | 8.1 | 35.0 | 28.7 |
| 18 | 14.4 | 8.0 | 34.0 | 31.2 |
| 19 | 17.0 | 7.7 | 34.9 | 41.1 |

**Seed 1**

| gen | B | C | D | E |
|---|---|---|---|---|
| 0 | 50.5 | 50.5 | 50.5 | 50.5 |
| 1 | 52.1 | 49.6 | 52.5 | 52.5 |
| 2 | 47.2 | 50.5 | 51.5 | 45.6 |
| 3 | 44.4 | 38.2 | 52.2 | 44.7 |
| 4 | 46.1 | 37.4 | 50.1 | 37.4 |
| 5 | 37.8 | 33.7 | 49.1 | 40.0 |
| 6 | 20.1 | 24.3 | 46.3 | 43.6 |
| 7 | 17.4 | 19.6 | 33.5 | 45.8 |
| 8 | 15.1 | 18.3 | 29.8 | 41.7 |
| 9 | 14.1 | 18.4 | 30.3 | 35.9 |
| 10 | 13.7 | 20.0 | 30.6 | 47.1 |
| 11 | 14.5 | 19.2 | 30.6 | 48.8 |
| 12 | 16.5 | 20.3 | 31.1 | 47.4 |
| 13 | 16.3 | 18.3 | 28.1 | 44.2 |
| 14 | 17.0 | 18.6 | 26.1 | 40.4 |
| 15 | 15.1 | 20.1 | 26.6 | 38.9 |
| 16 | 17.9 | 18.5 | 26.8 | 36.9 |
| 17 | 14.3 | 18.2 | 26.4 | 39.5 |
| 18 | 17.6 | 12.4 | 26.3 | 41.9 |
| 19 | 14.0 | 10.9 | 29.6 | 38.0 |

**Seed 2**

| gen | B | C | D | E |
|---|---|---|---|---|
| 0 | 53.0 | 53.0 | 53.0 | 53.0 |
| 1 | 51.3 | 54.6 | 54.7 | 54.7 |
| 2 | 32.5 | 47.7 | 51.3 | 50.0 |
| 3 | 24.4 | 40.9 | 52.5 | 49.5 |
| 4 | 24.0 | 25.1 | 46.9 | 53.6 |
| 5 | 18.7 | 19.2 | 46.0 | 53.0 |
| 6 | 18.7 | 15.5 | 41.3 | 49.7 |
| 7 | 26.6 | 10.4 | 36.6 | 40.4 |
| 8 | 23.9 | 9.4 | 35.5 | 44.4 |
| 9 | 28.9 | 12.1 | 31.1 | 40.6 |
| 10 | 25.0 | 10.7 | 36.3 | 43.7 |
| 11 | 24.1 | 10.5 | 37.7 | 40.3 |
| 12 | 30.2 | 10.8 | 38.1 | 39.2 |
| 13 | 33.4 | 10.4 | 39.3 | 38.5 |
| 14 | 35.2 | 11.1 | 39.4 | 38.1 |
| 15 | 33.3 | 11.8 | 38.0 | 41.6 |
| 16 | 21.8 | 12.9 | 36.8 | 36.9 |
| 17 | 19.0 | 13.8 | 33.3 | 40.4 |
| 18 | 15.4 | 10.7 | 36.1 | 31.7 |
| 19 | 16.8 | 8.8 | 32.0 | 33.0 |

**Seed 3**

| gen | B | C | D | E |
|---|---|---|---|---|
| 0 | 53.9 | 53.9 | 53.9 | 53.9 |
| 1 | 52.5 | 51.0 | 51.6 | 51.6 |
| 2 | 41.8 | 47.1 | 46.3 | 49.5 |
| 3 | 35.9 | 30.3 | 43.8 | 46.4 |
| 4 | 37.8 | 14.1 | 41.8 | 37.6 |
| 5 | 37.8 | 11.7 | 38.3 | 40.4 |
| 6 | 36.8 | 10.7 | 32.5 | 40.2 |
| 7 | 34.5 | 10.5 | 33.7 | 32.2 |
| 8 | 27.5 | 11.0 | 33.4 | 31.5 |
| 9 | 26.6 | 10.7 | 33.3 | 33.0 |
| 10 | 20.2 | 10.5 | 31.5 | 32.2 |
| 11 | 16.6 | 9.0 | 27.0 | 32.0 |
| 12 | 18.1 | 7.6 | 26.0 | 31.0 |
| 13 | 21.2 | 7.2 | 26.4 | 29.5 |
| 14 | 17.7 | 7.0 | 26.4 | 29.1 |
| 15 | 20.2 | 6.7 | 27.4 | 25.6 |
| 16 | 19.1 | 6.2 | 25.1 | 27.6 |
| 17 | 13.3 | 7.2 | 22.3 | 24.2 |
| 18 | 11.8 | 8.1 | 22.6 | 27.7 |
| 19 | 12.1 | 6.5 | 22.2 | 23.6 |

**Seed 4**

| gen | B | C | D | E |
|---|---|---|---|---|
| 0 | 52.2 | 52.2 | 52.2 | 52.2 |
| 1 | 47.8 | 50.9 | 51.7 | 51.8 |
| 2 | 47.0 | 48.8 | 49.4 | 48.9 |
| 3 | 45.9 | 44.7 | 48.9 | 48.7 |
| 4 | 34.2 | 36.8 | 47.7 | 50.1 |
| 5 | 33.9 | 28.6 | 48.5 | 48.5 |
| 6 | 28.4 | 27.5 | 47.2 | 42.6 |
| 7 | 20.9 | 27.2 | 48.5 | 44.9 |
| 8 | 22.7 | 27.1 | 45.2 | 41.5 |
| 9 | 18.6 | 27.5 | 45.7 | 45.3 |
| 10 | 17.7 | 27.7 | 44.5 | 36.9 |
| 11 | 9.8 | 26.5 | 47.2 | 33.3 |
| 12 | 9.2 | 26.2 | 44.0 | 34.9 |
| 13 | 10.5 | 23.1 | 42.0 | 32.1 |
| 14 | 14.4 | 15.8 | 42.0 | 31.2 |
| 15 | 13.4 | 9.6 | 40.2 | 43.2 |
| 16 | 14.5 | 9.8 | 38.2 | 40.2 |
| 17 | 13.9 | 8.9 | 42.7 | 37.1 |
| 18 | 12.9 | 9.1 | 28.8 | 40.8 |
| 19 | 15.1 | 10.2 | 36.7 | 34.9 |

### 9.4 Comparisons at the common budget

**D vs C**

| seed | n_cut | D best@n_cut | C best@n_cut | difference | higher |
|---|---|---|---|---|---|
| 0 | 278 | 0.3587 | 0.4632 | -0.1046 | C |
| 1 | 276 | 0.6372 | 0.4745 | +0.1627 | D |
| 2 | 280 | 0.4726 | 0.5138 | -0.0413 | C |
| 3 | 280 | 0.5807 | 0.5434 | +0.0373 | D |
| 4 | 275 | 0.4015 | 0.4370 | -0.0355 | C |

Mean difference +0.0037 (range -0.1046 to +0.1627). **NO DEMONSTRATED DIFFERENCE between D and C: D higher in 2, lower in 3, tied in 0 of 5 seeds.** Smallest two-sided sign-test p possible with 5 seeds: 0.0625.

**D vs E**

| seed | n_cut | D best@n_cut | E best@n_cut | difference | higher |
|---|---|---|---|---|---|
| 0 | 278 | 0.3587 | 0.4164 | -0.0577 | E |
| 1 | 276 | 0.6372 | 0.4719 | +0.1653 | D |
| 2 | 280 | 0.4726 | 0.4773 | -0.0047 | E |
| 3 | 280 | 0.5807 | 0.6099 | -0.0292 | E |
| 4 | 275 | 0.4015 | 0.4486 | -0.0471 | E |

Mean difference +0.0053 (range -0.0577 to +0.1653). **NO DEMONSTRATED DIFFERENCE between D and E: D higher in 1, lower in 4, tied in 0 of 5 seeds.** Smallest two-sided sign-test p possible with 5 seeds: 0.0625.

**E vs C**

| seed | n_cut | E best@n_cut | C best@n_cut | difference | higher |
|---|---|---|---|---|---|
| 0 | 278 | 0.4164 | 0.4632 | -0.0469 | C |
| 1 | 276 | 0.4719 | 0.4745 | -0.0026 | C |
| 2 | 280 | 0.4773 | 0.5138 | -0.0366 | C |
| 3 | 280 | 0.6099 | 0.5434 | +0.0665 | E |
| 4 | 275 | 0.4486 | 0.4370 | +0.0115 | E |

Mean difference -0.0016 (range -0.0469 to +0.0665). **NO DEMONSTRATED DIFFERENCE between E and C: E higher in 2, lower in 3, tied in 0 of 5 seeds.** Smallest two-sided sign-test p possible with 5 seeds: 0.0625.

### 9.5 Does D's extra diversity, if any, convert into fitness?

| seed | diversity D-C (mean gens 1-19) | D-C (gen 19) | diversity D-E (mean gens 1-19) | D-E (gen 19) | best TM D-C @n_cut | best TM D-E @n_cut |
|---|---|---|---|---|---|---|
| 0 | +21.6 | +27.1 | +1.8 | -6.2 | -0.1046 | -0.0577 |
| 1 | +11.1 | +18.7 | -7.0 | -8.4 | +0.1627 | +0.1653 |
| 2 | +21.9 | +23.2 | -3.0 | -1.0 | -0.0413 | -0.0047 |
| 3 | +17.8 | +15.7 | -1.8 | -1.4 | +0.0373 | -0.0292 |
| 4 | +18.6 | +26.6 | +2.8 | +1.9 | -0.0355 | -0.0471 |

Diversity differences are in edit-distance units (positive = D more diverse); fitness differences in TM-score at n_cut (positive = D higher). Population diversity is over the evaluated population (16-20 genomes).

### 9.6 The injected genomes themselves (D: circle proposals; E: random immigrants)

| seed | D injected: mean TM | D injected: best | D rest of population: mean | E injected: mean TM | E injected: best | E rest of population: mean |
|---|---|---|---|---|---|---|
| 0 | 0.2732 | 0.3321 | 0.2983 | 0.2480 | 0.4599 | 0.3113 |
| 1 | 0.2706 | 0.3142 | 0.3791 | 0.2552 | 0.4487 | 0.3256 |
| 2 | 0.2493 | 0.3568 | 0.3117 | 0.2569 | 0.3696 | 0.3302 |
| 3 | 0.2538 | 0.3144 | 0.3898 | 0.2514 | 0.3540 | 0.4029 |
| 4 | 0.2589 | 0.3775 | 0.3038 | 0.2390 | 0.3838 | 0.3232 |

Means over generations 1-19 of the last 4 members of each evaluated population (the injected ones) and of the other members.

### 9.7 LLM calls, tokens and wall time (seconds; percentages of that run's wall time)

| seed | arm | LLM calls | requests | tokens in | tokens out | wall | fitness (folds) | LLM calls time | swap between models |   of which Ollama load | everything else |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | C GA (LLM ops) | 390 | 399 | 147,500 | 10,522 | 2875 | 614 (21.3%) | 1950 (67.8%) | 298 (10.4%) | in LLM time | 13.7 (0.48%) |
| 0 | D GA + LLM + circles | 576 | 620 | 250,945 | 19,060 | 4657 | 784 (16.8%) | 3427 (73.6%) | 431 (9.3%) | 185 | 14.8 (0.32%) |
| 0 | E GA + LLM + random immigrants | 383 | 397 | 146,862 | 10,534 | 3139 | 836 (26.6%) | 1827 (58.2%) | 463 (14.8%) | 192 | 11.8 (0.38%) |
| 1 | C GA (LLM ops) | 390 | 393 | 147,446 | 10,464 | 3013 | 654 (21.7%) | 2056 (68.2%) | 291 (9.7%) | in LLM time | 11.5 (0.38%) |
| 1 | D GA + LLM + circles | 577 | 625 | 262,380 | 19,694 | 4909 | 912 (18.6%) | 3511 (71.5%) | 472 (9.6%) | 196 | 13.0 (0.26%) |
| 1 | E GA + LLM + random immigrants | 385 | 397 | 148,643 | 10,787 | 3222 | 899 (27.9%) | 1863 (57.8%) | 446 (13.8%) | 184 | 14.1 (0.44%) |
| 2 | C GA (LLM ops) | 383 | 411 | 157,273 | 13,593 | 3859 | 909 (23.5%) | 2664 (69.0%) | 276 (7.1%) | in LLM time | 11.2 (0.29%) |
| 2 | D GA + LLM + circles | 578 | 606 | 245,092 | 18,181 | 4808 | 975 (20.3%) | 3386 (70.4%) | 433 (9.0%) | 200 | 13.6 (0.28%) |
| 2 | E GA + LLM + random immigrants | 382 | 404 | 155,346 | 13,170 | 3933 | 1121 (28.5%) | 2368 (60.2%) | 431 (11.0%) | 205 | 13.4 (0.34%) |
| 3 | C GA (LLM ops) | 389 | 406 | 149,333 | 10,820 | 3027 | 623 (20.6%) | 2120 (70.0%) | 271 (9.0%) | in LLM time | 13.0 (0.43%) |
| 3 | D GA + LLM + circles | 582 | 615 | 245,049 | 17,949 | 4783 | 816 (17.1%) | 3481 (72.8%) | 472 (9.9%) | 210 | 14.5 (0.30%) |
| 3 | E GA + LLM + random immigrants | 387 | 402 | 148,829 | 10,731 | 3250 | 851 (26.2%) | 1903 (58.5%) | 484 (14.9%) | 236 | 13.1 (0.40%) |
| 4 | C GA (LLM ops) | 388 | 403 | 153,927 | 11,592 | 3318 | 839 (25.3%) | 2218 (66.9%) | 247 (7.4%) | in LLM time | 13.7 (0.41%) |
| 4 | D GA + LLM + circles | 577 | 610 | 250,176 | 18,869 | 4922 | 936 (19.0%) | 3479 (70.7%) | 492 (10.0%) | 254 | 15.2 (0.31%) |
| 4 | E GA + LLM + random immigrants | 389 | 402 | 153,564 | 11,867 | 3491 | 986 (28.2%) | 2069 (59.3%) | 422 (12.1%) | 179 | 14.1 (0.40%) |

Swap = unloading Ollama, moving ESMFold between CPU and GPU, and (D, E) loading the Ollama model, timed separately from the LLM calls. **Arm C was run with different accounting:** its swap time excludes the Ollama reload, which fell inside the first LLM call of each breeding step and is therefore in its LLM time. Estimate of that reload for C (excess latency of the first request after each fold phase over the operator median, from its call log): seed 0: 175 s, seed 1: 139 s, seed 2: 140 s, seed 3: 164 s, seed 4: 140 s. D and E carry the measured load in swap time.

### 9.8 Arm D, per operator: calls, fallbacks, requests per call

| seed | operator | operator calls | LLM calls | fell back | fallback rate | requests / call | wall s |
|---|---|---|---|---|---|---|---|
| 0 | mutate | 266 | 266 | 0 | 0.000 | 1.068 | 1246 |
| 0 | crossover | 118 | 118 | 0 | 0.000 | 1.000 | 595 |
| 0 | propose | 76 | 76 | 9 | 0.118 | 1.342 | 615 |
| 0 | observe | 68 | 68 | 0 | 0.000 | 1.000 | 383 |
| 0 | consult | 38 | 38 | 0 | 0.000 | 1.000 | 215 |
| 0 | central_directive | 4 | 4 | 0 | 0.000 | 1.000 | 28 |
| 0 | curate | 6 | 6 | 0 | 0.000 | 1.000 | 346 |
| 1 | mutate | 266 | 266 | 0 | 0.000 | 1.030 | 1188 |
| 1 | crossover | 119 | 119 | 0 | 0.000 | 1.000 | 623 |
| 1 | propose | 76 | 76 | 14 | 0.184 | 1.526 | 727 |
| 1 | observe | 68 | 68 | 0 | 0.000 | 1.000 | 373 |
| 1 | consult | 38 | 38 | 0 | 0.000 | 1.000 | 202 |
| 1 | central_directive | 4 | 4 | 0 | 0.000 | 1.000 | 20 |
| 1 | curate | 6 | 6 | 0 | 0.000 | 1.000 | 379 |
| 2 | mutate | 266 | 266 | 0 | 0.000 | 1.041 | 1271 |
| 2 | crossover | 120 | 120 | 0 | 0.000 | 1.000 | 642 |
| 2 | propose | 76 | 76 | 3 | 0.039 | 1.197 | 515 |
| 2 | observe | 68 | 68 | 0 | 0.000 | 1.000 | 378 |
| 2 | consult | 38 | 38 | 0 | 0.000 | 1.000 | 200 |
| 2 | central_directive | 4 | 4 | 0 | 0.000 | 1.000 | 22 |
| 2 | curate | 6 | 6 | 1 | 0.167 | 1.333 | 359 |
| 3 | mutate | 266 | 266 | 0 | 0.000 | 1.053 | 1304 |
| 3 | crossover | 124 | 124 | 0 | 0.000 | 1.000 | 685 |
| 3 | propose | 76 | 76 | 2 | 0.026 | 1.250 | 520 |
| 3 | observe | 68 | 68 | 0 | 0.000 | 1.000 | 406 |
| 3 | consult | 38 | 38 | 0 | 0.000 | 1.000 | 214 |
| 3 | central_directive | 4 | 4 | 0 | 0.000 | 1.000 | 29 |
| 3 | curate | 6 | 6 | 0 | 0.000 | 1.000 | 322 |
| 4 | mutate | 266 | 266 | 0 | 0.000 | 1.041 | 1315 |
| 4 | crossover | 119 | 119 | 0 | 0.000 | 1.000 | 592 |
| 4 | propose | 76 | 76 | 7 | 0.092 | 1.289 | 599 |
| 4 | observe | 68 | 68 | 0 | 0.000 | 1.000 | 386 |
| 4 | consult | 38 | 38 | 0 | 0.000 | 1.000 | 214 |
| 4 | central_directive | 4 | 4 | 0 | 0.000 | 1.000 | 27 |
| 4 | curate | 6 | 6 | 0 | 0.000 | 1.000 | 346 |

**Circle proposal fallback rate** (`propose`, position edits to a population member) is the `propose` rows; the stage-1 probe gave 1 of 50. `observe`, `consult`, `central_directive` and `curate` fall back to nothing (no observation written, empty note, deterministic directive, board unchanged).

Over all 5 completed D runs (seeds [0, 1, 2, 3, 4]): propose 35/380; observe 0/340; consult 0/190; central_directive 0/20; curate 1/30; mutate 0/1330; crossover 0/600 (fell back / LLM calls).

Blackboard at the end of each D run:

| seed | entries written | live at end | observations | directives | summaries (curation) |
|---|---|---|---|---|---|
| 0 | 80 | 9 | 68 | 4 | 8 |
| 1 | 80 | 9 | 68 | 4 | 8 |
| 2 | 79 | 8 | 68 | 4 | 7 |
| 3 | 80 | 23 | 68 | 4 | 8 |
| 4 | 82 | 9 | 68 | 4 | 10 |

### 9.9 Edit distance of the best genomes to the 63-residue reference sequence

| seed | arm | best TM @n_cut | edit dist to reference | length | final best TM (own budget) | edit dist | length |
|---|---|---|---|---|---|---|---|
| 0 | A random search | 0.3859 | 55 | 50 | 0.3859 | 55 | 50 |
| 0 | B GA (deterministic ops) | 0.4203 | 55 | 59 | 0.4203 | 55 | 59 |
| 0 | C GA (LLM ops) | 0.4632 | 54 | 59 | 0.4632 | 54 | 59 |
| 0 | D GA + LLM + circles | 0.3587 | 54 | 52 | 0.3639 | 57 | 61 |
| 0 | E GA + LLM + random immigrants | 0.4164 | 55 | 62 | 0.4599 | 61 | 71 |
| 1 | A random search | 0.4004 | 67 | 79 | 0.4004 | 67 | 79 |
| 1 | B GA (deterministic ops) | 0.5025 | 55 | 64 | 0.5025 | 55 | 64 |
| 1 | C GA (LLM ops) | 0.4745 | 54 | 62 | 0.4745 | 54 | 62 |
| 1 | D GA + LLM + circles | 0.6372 | 54 | 66 | 0.6372 | 54 | 66 |
| 1 | E GA + LLM + random immigrants | 0.4719 | 56 | 62 | 0.4848 | 53 | 65 |
| 2 | A random search | 0.4104 | 58 | 66 | 0.4104 | 58 | 66 |
| 2 | B GA (deterministic ops) | 0.5040 | 56 | 72 | 0.5040 | 56 | 72 |
| 2 | C GA (LLM ops) | 0.5138 | 59 | 72 | 0.5138 | 59 | 72 |
| 2 | D GA + LLM + circles | 0.4726 | 57 | 69 | 0.4726 | 57 | 69 |
| 2 | E GA + LLM + random immigrants | 0.4773 | 60 | 74 | 0.4787 | 59 | 74 |
| 3 | A random search | 0.4035 | 55 | 59 | 0.4035 | 55 | 59 |
| 3 | B GA (deterministic ops) | 0.4007 | 63 | 70 | 0.4007 | 63 | 70 |
| 3 | C GA (LLM ops) | 0.5434 | 52 | 57 | 0.5434 | 52 | 57 |
| 3 | D GA + LLM + circles | 0.5807 | 53 | 57 | 0.5807 | 53 | 57 |
| 3 | E GA + LLM + random immigrants | 0.6099 | 53 | 59 | 0.6146 | 52 | 59 |
| 4 | A random search | 0.4009 | 55 | 62 | 0.4009 | 55 | 62 |
| 4 | B GA (deterministic ops) | 0.4873 | 59 | 62 | 0.4873 | 59 | 62 |
| 4 | C GA (LLM ops) | 0.4370 | 59 | 69 | 0.4370 | 59 | 69 |
| 4 | D GA + LLM + circles | 0.4015 | 55 | 63 | 0.4034 | 55 | 64 |
| 4 | E GA + LLM + random immigrants | 0.4486 | 60 | 69 | 0.4486 | 60 | 69 |

### 9.10 What the circle proposals did (from the D call logs; accepted responses only)

| seed | accepted proposals | positions changed | distinct position indices | most-used index (share) | distinct new letters | most-used letter (share) | slot proposals that beat their base (gens 2-19) |
|---|---|---|---|---|---|---|---|
| 0 | 67 | 199 | 50 | 15 (7%) | 16 | Y (24%) | 31/72 |
| 1 | 62 | 181 | 51 | 33 (6%) | 16 | L (25%) | 31/72 |
| 2 | 73 | 182 | 43 | 23 (8%) | 17 | L (31%) | 38/72 |
| 3 | 74 | 166 | 40 | 14 (8%) | 14 | L (22%) | 31/72 |
| 4 | 69 | 188 | 46 | 25 (9%) | 18 | L (29%) | 35/72 |

