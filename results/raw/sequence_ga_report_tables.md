### 1. Runs

| seed | arm | distinct evaluations | cache hits | final best TM | final best length | edit distance to reference | attempt |
|---|---|---|---|---|---|---|---|
| 0 | A random search | 282 | 0 | 0.3859 | 50 | 55 | 1 |
| 0 | B GA (deterministic ops) | 278 | 42 | 0.4203 | 59 | 55 | 1 |
| 0 | C GA (LLM ops) | 282 | 38 | 0.4632 | 59 | 54 | 1 |
| 1 | A random search | 280 | 0 | 0.4004 | 79 | 67 | 1 |
| 1 | B GA (deterministic ops) | 276 | 44 | 0.5025 | 64 | 55 | 1 |
| 1 | C GA (LLM ops) | 280 | 40 | 0.4745 | 62 | 54 | 1 |
| 2 | A random search | 281 | 0 | 0.4104 | 66 | 58 | 1 |
| 2 | B GA (deterministic ops) | 280 | 40 | 0.5040 | 72 | 56 | 1 |
| 2 | C GA (LLM ops) | 281 | 39 | 0.5138 | 72 | 59 | 1 |
| 3 | A random search | 280 | 0 | 0.4035 | 59 | 55 | 1 |
| 3 | B GA (deterministic ops) | 280 | 40 | 0.4007 | 70 | 63 | 1 |
| 3 | C GA (LLM ops) | 280 | 40 | 0.5434 | 57 | 52 | 1 |
| 4 | A random search | 280 | 0 | 0.4009 | 62 | 55 | 1 |
| 4 | B GA (deterministic ops) | 275 | 45 | 0.4873 | 62 | 59 | 1 |
| 4 | C GA (LLM ops) | 280 | 40 | 0.4370 | 69 | 59 | 1 |

### 2. Best-so-far TM-score vs distinct evaluations (per seed, never pooled)

**Seed 0**

| distinct evaluations | A random search | B GA (deterministic ops) | C GA (LLM ops) |
|---|---|---|---|
| 16 | 0.3006 | 0.3006 | 0.3006 |
| 32 | 0.3081 | 0.3150 | 0.3287 |
| 64 | 0.3824 | 0.3392 | 0.3734 |
| 96 | 0.3824 | 0.3605 | 0.3734 |
| 128 | 0.3824 | 0.4003 | 0.4072 |
| 160 | 0.3859 | 0.4003 | 0.4454 |
| 192 | 0.3859 | 0.4003 | 0.4504 |
| 224 | 0.3859 | 0.4203 | 0.4560 |
| 256 | 0.3859 | 0.4203 | 0.4560 |
| own final (n) | 0.3859 (282) | 0.4203 (278) | 0.4632 (282) |

**Seed 1**

| distinct evaluations | A random search | B GA (deterministic ops) | C GA (LLM ops) |
|---|---|---|---|
| 16 | 0.2959 | 0.2959 | 0.2959 |
| 32 | 0.3838 | 0.3871 | 0.4227 |
| 64 | 0.3838 | 0.3871 | 0.4272 |
| 96 | 0.3838 | 0.4170 | 0.4272 |
| 128 | 0.3838 | 0.4367 | 0.4457 |
| 160 | 0.4004 | 0.4782 | 0.4745 |
| 192 | 0.4004 | 0.4782 | 0.4745 |
| 224 | 0.4004 | 0.4843 | 0.4745 |
| 256 | 0.4004 | 0.4843 | 0.4745 |
| own final (n) | 0.4004 (280) | 0.5025 (276) | 0.4745 (280) |

**Seed 2**

| distinct evaluations | A random search | B GA (deterministic ops) | C GA (LLM ops) |
|---|---|---|---|
| 16 | 0.3353 | 0.3353 | 0.3353 |
| 32 | 0.3353 | 0.3492 | 0.3433 |
| 64 | 0.3629 | 0.4074 | 0.3833 |
| 96 | 0.3629 | 0.4175 | 0.3908 |
| 128 | 0.4094 | 0.4386 | 0.4174 |
| 160 | 0.4094 | 0.4821 | 0.4732 |
| 192 | 0.4094 | 0.4821 | 0.5004 |
| 224 | 0.4094 | 0.4821 | 0.5004 |
| 256 | 0.4094 | 0.5040 | 0.5138 |
| own final (n) | 0.4104 (281) | 0.5040 (280) | 0.5138 (281) |

**Seed 3**

| distinct evaluations | A random search | B GA (deterministic ops) | C GA (LLM ops) |
|---|---|---|---|
| 16 | 0.2983 | 0.2983 | 0.2983 |
| 32 | 0.3320 | 0.3572 | 0.4124 |
| 64 | 0.3320 | 0.3572 | 0.4319 |
| 96 | 0.3605 | 0.3820 | 0.4696 |
| 128 | 0.4035 | 0.3820 | 0.5434 |
| 160 | 0.4035 | 0.3820 | 0.5434 |
| 192 | 0.4035 | 0.3820 | 0.5434 |
| 224 | 0.4035 | 0.3820 | 0.5434 |
| 256 | 0.4035 | 0.4007 | 0.5434 |
| own final (n) | 0.4035 (280) | 0.4007 (280) | 0.5434 (280) |

**Seed 4**

| distinct evaluations | A random search | B GA (deterministic ops) | C GA (LLM ops) |
|---|---|---|---|
| 16 | 0.3447 | 0.3447 | 0.3447 |
| 32 | 0.3447 | 0.3727 | 0.3777 |
| 64 | 0.3756 | 0.4080 | 0.3777 |
| 96 | 0.4009 | 0.4263 | 0.4104 |
| 128 | 0.4009 | 0.4506 | 0.4104 |
| 160 | 0.4009 | 0.4691 | 0.4104 |
| 192 | 0.4009 | 0.4835 | 0.4104 |
| 224 | 0.4009 | 0.4873 | 0.4370 |
| 256 | 0.4009 | 0.4873 | 0.4370 |
| own final (n) | 0.4009 (280) | 0.4873 (275) | 0.4370 (280) |

### 3. Final best per seed; then mean and range over seeds

| arm | seed 0 | seed 1 | seed 2 | seed 3 | seed 4 |
|---|---|---|---|---|---|
| A random search (own budget) | 0.3859 | 0.4004 | 0.4104 | 0.4035 | 0.4009 |
| B GA (deterministic ops) (own budget) | 0.4203 | 0.5025 | 0.5040 | 0.4007 | 0.4873 |
| C GA (LLM ops) (own budget) | 0.4632 | 0.4745 | 0.5138 | 0.5434 | 0.4370 |
| A read at B's distinct-evaluation count | 0.3859 | 0.4004 | 0.4104 | 0.4035 | 0.4009 |
| A read at C's distinct-evaluation count | 0.3859 | 0.4004 | 0.4104 | 0.4035 | 0.4009 |

Cross-seed summary (the only pooled table; n = number of seeds that finished):

| arm | n seeds | mean final best | range (min – max) |
|---|---|---|---|
| A (own budget) | 5 | 0.4002 | 0.3859 – 0.4104 |
| A at B's budget | 5 | 0.4002 | 0.3859 – 0.4104 |
| A at C's budget | 5 | 0.4002 | 0.3859 – 0.4104 |
| B | 5 | 0.4630 | 0.4007 – 0.5040 |
| C | 5 | 0.4864 | 0.4370 – 0.5434 |

### 4. Mean pairwise edit distance within the population, per generation

| gen | B s0 | B s1 | B s2 | B s3 | B s4 | C s0 | C s1 | C s2 | C s3 | C s4 |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 50.3 | 50.5 | 53.0 | 53.9 | 52.2 | 50.3 | 50.5 | 53.0 | 53.9 | 52.2 |
| 1 | 42.0 | 52.1 | 51.3 | 52.5 | 47.8 | 42.4 | 49.6 | 54.6 | 51.0 | 50.9 |
| 2 | 40.9 | 47.2 | 32.5 | 41.8 | 47.0 | 38.9 | 50.5 | 47.7 | 47.1 | 48.8 |
| 3 | 38.2 | 44.4 | 24.4 | 35.9 | 45.9 | 26.4 | 38.2 | 40.9 | 30.3 | 44.7 |
| 4 | 31.1 | 46.1 | 24.0 | 37.8 | 34.2 | 27.5 | 37.4 | 25.1 | 14.1 | 36.8 |
| 5 | 18.6 | 37.8 | 18.7 | 37.8 | 33.9 | 28.5 | 33.7 | 19.2 | 11.7 | 28.6 |
| 6 | 14.3 | 20.1 | 18.7 | 36.8 | 28.4 | 28.8 | 24.3 | 15.5 | 10.7 | 27.5 |
| 7 | 14.8 | 17.4 | 26.6 | 34.5 | 20.9 | 21.1 | 19.6 | 10.4 | 10.5 | 27.2 |
| 8 | 15.5 | 15.1 | 23.9 | 27.5 | 22.7 | 18.3 | 18.3 | 9.4 | 11.0 | 27.1 |
| 9 | 14.2 | 14.1 | 28.9 | 26.6 | 18.6 | 17.1 | 18.4 | 12.1 | 10.7 | 27.5 |
| 10 | 15.8 | 13.7 | 25.0 | 20.2 | 17.7 | 10.0 | 20.0 | 10.7 | 10.5 | 27.7 |
| 11 | 14.1 | 14.5 | 24.1 | 16.6 | 9.8 | 9.2 | 19.2 | 10.5 | 9.0 | 26.5 |
| 12 | 16.8 | 16.5 | 30.2 | 18.1 | 9.2 | 8.0 | 20.3 | 10.8 | 7.6 | 26.2 |
| 13 | 14.4 | 16.3 | 33.4 | 21.2 | 10.5 | 7.5 | 18.3 | 10.4 | 7.2 | 23.1 |
| 14 | 18.4 | 17.0 | 35.2 | 17.7 | 14.4 | 7.5 | 18.6 | 11.1 | 7.0 | 15.8 |
| 15 | 18.4 | 15.1 | 33.3 | 20.2 | 13.4 | 9.0 | 20.1 | 11.8 | 6.7 | 9.6 |
| 16 | 17.6 | 17.9 | 21.8 | 19.1 | 14.5 | 8.1 | 18.5 | 12.9 | 6.2 | 9.8 |
| 17 | 14.6 | 14.3 | 19.0 | 13.3 | 13.9 | 8.1 | 18.2 | 13.8 | 7.2 | 8.9 |
| 18 | 14.4 | 17.6 | 15.4 | 11.8 | 12.9 | 8.0 | 12.4 | 10.7 | 8.1 | 9.1 |
| 19 | 17.0 | 14.0 | 16.8 | 12.1 | 15.1 | 7.7 | 10.9 | 8.8 | 6.5 | 10.2 |

Distinct genomes in the population (of 16), same columns:

| gen | B s0 | B s1 | B s2 | B s3 | B s4 | C s0 | C s1 | C s2 | C s3 | C s4 |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 |
| 1 | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 |
| 2 | 15 | 16 | 16 | 15 | 16 | 16 | 16 | 16 | 16 | 16 |
| 3 | 15 | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 |
| 4 | 15 | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 |
| 5 | 14 | 16 | 16 | 16 | 16 | 16 | 16 | 15 | 16 | 16 |
| 6 | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 15 | 16 | 16 |
| 7 | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 |
| 8 | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 15 | 16 |
| 9 | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 |
| 10 | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 |
| 11 | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 |
| 12 | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 |
| 13 | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 |
| 14 | 16 | 14 | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 |
| 15 | 16 | 15 | 16 | 16 | 16 | 16 | 15 | 16 | 16 | 16 |
| 16 | 16 | 15 | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 |
| 17 | 15 | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 |
| 18 | 16 | 16 | 15 | 16 | 15 | 16 | 16 | 16 | 16 | 16 |
| 19 | 16 | 16 | 15 | 15 | 15 | 16 | 16 | 16 | 16 | 16 |

### 5. Final best genomes and their edit distance to the reference sequence

| seed | arm | TM | length | edit distance to reference (63 aa) |
|---|---|---|---|---|
| 0 | A random search | 0.3859 | 50 | 55 |
| 0 | B GA (deterministic ops) | 0.4203 | 59 | 55 |
| 0 | C GA (LLM ops) | 0.4632 | 59 | 54 |
| 1 | A random search | 0.4004 | 79 | 67 |
| 1 | B GA (deterministic ops) | 0.5025 | 64 | 55 |
| 1 | C GA (LLM ops) | 0.4745 | 62 | 54 |
| 2 | A random search | 0.4104 | 66 | 58 |
| 2 | B GA (deterministic ops) | 0.5040 | 72 | 56 |
| 2 | C GA (LLM ops) | 0.5138 | 72 | 59 |
| 3 | A random search | 0.4035 | 59 | 55 |
| 3 | B GA (deterministic ops) | 0.4007 | 70 | 63 |
| 3 | C GA (LLM ops) | 0.5434 | 57 | 52 |
| 4 | A random search | 0.4009 | 62 | 55 |
| 4 | B GA (deterministic ops) | 0.4873 | 62 | 59 |
| 4 | C GA (LLM ops) | 0.4370 | 69 | 59 |

```
seed 0 A: IKRGQHVTVRGSFSSVNYTCHGDMPSPYTPCWMFGKGWGREYNYYIVYPM
seed 0 B: CSKKFVALYAVVKFISNCLNGYFLPQANSHYGIFIQWQCPWQCGRDKGNWSVIKCRCQN
seed 0 C: LASETGARDMGVVFIMNYLNWYFAYKMGITVCHKGFNWCYSGQNVRHPWLAFFKMMNDM
seed 1 A: GCMAMGFNIWCQGMWHISLKWTTWTSIGDWIRFPGAECCVCYSRASKLKPSEDLPQHITFDLHENKFINNGEEHLIDCM
seed 1 B: LPWMIPSCDTDWLIKQVYWADGTVDKKDSYTEITKDGTNWTVCGLVMNYIDEPGELYMIKSIAY
seed 1 C: RDGSVPLCMGEWFIMQVYWYDIIAIKDHTHIMKDDTNRTVHGLVKWYIPVGGSKYMIKYIAY
seed 2 A: PMCMFSPDKHSFFAHYGGGQMIIVWFITTVGKSYSTVIPSTGCWEHWWGPWVTPFQPPAIILHHLY
seed 2 B: WEARSTWTSTQFMKKYYQVSCCCLGGERITCIIRDKDWILYQKQKFYQCPMGITDVEAIGAEHEVRERVPLH
seed 2 C: MEDRVTWPSTSMYKKGQAVFGKCLGNERITGIISDGDWIYRNKQKMAFCPQGETDIEEKGGEHATRRLVPHH
seed 3 A: KKSQCTLRQGDDWTLWEVKNRLHVFHGHAMHEFQAPSGCSVFNDNPPNKGWTFTWSIHS
seed 3 B: RMQCCIPCPSEVFWDMWVWGYWTKHSMKSPHGWNIMSFQNYHRWVASDPCRIILHKIHKFGYCKGCMGQP
seed 3 C: NYLSGWCANKGLWYRTNGMNYHKWWTKSPEQWFLTIKIMGQTEYMVIRGDMHWRKIE
seed 4 A: GQEGDIENWLCVVCYWQGGYMFGGWVLIFEITGPPEQASGKYPLWDNIHNQYWNQCFFIKVL
seed 4 B: IWFYCFGYRWHLEEMLWYFDGAGVKNWMMWYNGRYMEFDRADMYFKCYMVSRVWIMSPCHTN
seed 4 C: MAGEKALQRGIMVIDSRKTVTMFHDGHNRKNWKEPGIVCHGYMVSMGADCYWSVKYFCNDTALNDDVRP
```

### 6. Wall time split (seconds; percentages of that run's wall time)

| seed | arm | wall | fitness (folds) | LLM calls | GPU swap (C only) | everything else | s per distinct fold |
|---|---|---|---|---|---|---|---|
| 0 | A random search | 641 | 641 (100.0%) | 0 (0.0%) | 0 (0.0%) | 0.0 (0.00%) | 2.27 |
| 0 | B GA (deterministic ops) | 585 | 583 (99.5%) | 0 (0.0%) | 0 (0.0%) | 2.7 (0.46%) | 2.10 |
| 0 | C GA (LLM ops) | 2875 | 614 (21.3%) | 1950 (67.8%) | 298 (10.4%) | 13.7 (0.48%) | 2.18 |
| 1 | A random search | 658 | 658 (100.0%) | 0 (0.0%) | 0 (0.0%) | 0.0 (0.00%) | 2.35 |
| 1 | B GA (deterministic ops) | 683 | 681 (99.6%) | 0 (0.0%) | 0 (0.0%) | 2.7 (0.39%) | 2.47 |
| 1 | C GA (LLM ops) | 3013 | 654 (21.7%) | 2056 (68.2%) | 291 (9.7%) | 11.5 (0.38%) | 2.34 |
| 2 | A random search | 689 | 689 (100.0%) | 0 (0.0%) | 0 (0.0%) | 0.0 (0.00%) | 2.45 |
| 2 | B GA (deterministic ops) | 937 | 933 (99.6%) | 0 (0.0%) | 0 (0.0%) | 3.6 (0.39%) | 3.33 |
| 2 | C GA (LLM ops) | 3859 | 909 (23.5%) | 2664 (69.0%) | 276 (7.1%) | 11.2 (0.29%) | 3.23 |
| 3 | A random search | 614 | 614 (100.0%) | 0 (0.0%) | 0 (0.0%) | 0.0 (0.00%) | 2.19 |
| 3 | B GA (deterministic ops) | 762 | 759 (99.6%) | 0 (0.0%) | 0 (0.0%) | 3.1 (0.41%) | 2.71 |
| 3 | C GA (LLM ops) | 3027 | 623 (20.6%) | 2120 (70.0%) | 271 (9.0%) | 13.0 (0.43%) | 2.22 |
| 4 | A random search | 595 | 595 (100.0%) | 0 (0.0%) | 0 (0.0%) | 0.0 (0.00%) | 2.13 |
| 4 | B GA (deterministic ops) | 612 | 609 (99.6%) | 0 (0.0%) | 0 (0.0%) | 2.5 (0.41%) | 2.22 |
| 4 | C GA (LLM ops) | 3318 | 839 (25.3%) | 2218 (66.9%) | 247 (7.4%) | 13.7 (0.41%) | 3.00 |

LLM time is time inside the two operator wrappers: retries and the Ollama model reload that follows each swap are included. "Everything else" is selection/crossover/mutation bookkeeping, diversity metrics and logging.

### 7. Arm C: fallback rate and requests per call

| seed | operator | operator calls | gated off (crossover rate) | LLM calls | fell back | fallback rate | requests / LLM call | retries |
|---|---|---|---|---|---|---|---|---|
| 0 | mutate (position) | 266 | 0 | 266 | 0 | 0.000 | 1.034 | 9 |
| 0 | crossover (segment) | 133 | 9 | 124 | 0 | 0.000 | 1.000 | 0 |
| 0 | both |  |  | 390 | 0 | 0.000 | 1.023 |  |
| 1 | mutate (position) | 266 | 0 | 266 | 0 | 0.000 | 1.011 | 3 |
| 1 | crossover (segment) | 133 | 9 | 124 | 0 | 0.000 | 1.000 | 0 |
| 1 | both |  |  | 390 | 0 | 0.000 | 1.008 |  |
| 2 | mutate (position) | 266 | 0 | 266 | 0 | 0.000 | 1.105 | 28 |
| 2 | crossover (segment) | 133 | 16 | 117 | 0 | 0.000 | 1.000 | 0 |
| 2 | both |  |  | 383 | 0 | 0.000 | 1.073 |  |
| 3 | mutate (position) | 266 | 0 | 266 | 1 | 0.004 | 1.064 | 18 |
| 3 | crossover (segment) | 133 | 10 | 123 | 0 | 0.000 | 1.000 | 0 |
| 3 | both |  |  | 389 | 1 | 0.003 | 1.044 |  |
| 4 | mutate (position) | 266 | 0 | 266 | 0 | 0.000 | 1.056 | 15 |
| 4 | crossover (segment) | 133 | 11 | 122 | 0 | 0.000 | 1.000 | 0 |
| 4 | both |  |  | 388 | 0 | 0.000 | 1.039 |  |

Fallback rate = fell back / LLM calls (a crossover the 0.9 rate gate skipped never reaches the LLM and is excluded); requests / LLM call is 1.0 when every call succeeded first time.

Stage-1 gate (one generation, pop 16, seed 0): fallback rate by operator {'mutate': 0.0, 'crossover': 0.0}; LLM calls {'mutate': 14, 'crossover': 7}.

### 8. Comparisons at equal distinct-evaluation budget

**B vs A**

| seed | budget n | B best@n | A best@n | difference | higher |
|---|---|---|---|---|---|
| 0 | 278 | 0.4203 | 0.3859 | +0.0343 | B |
| 1 | 276 | 0.5025 | 0.4004 | +0.1021 | B |
| 2 | 280 | 0.5040 | 0.4104 | +0.0936 | B |
| 3 | 280 | 0.4007 | 0.4035 | -0.0028 | A |
| 4 | 275 | 0.4873 | 0.4009 | +0.0864 | B |

Mean difference +0.0627 (range -0.0028 to +0.1021). **NO DEMONSTRATED DIFFERENCE between B and A: B higher in 4, lower in 1, tied in 0 of 5 seeds.** Smallest two-sided sign-test p possible with 5 seeds: 0.0625.

**C vs A**

| seed | budget n | C best@n | A best@n | difference | higher |
|---|---|---|---|---|---|
| 0 | 282 | 0.4632 | 0.3859 | +0.0773 | C |
| 1 | 280 | 0.4745 | 0.4004 | +0.0741 | C |
| 2 | 281 | 0.5138 | 0.4104 | +0.1034 | C |
| 3 | 280 | 0.5434 | 0.4035 | +0.1399 | C |
| 4 | 280 | 0.4370 | 0.4009 | +0.0361 | C |

Mean difference +0.0862 (range +0.0361 to +0.1399). **C BEATS A under the pre-set rule (higher in all 5 of 5 seeds).** Smallest two-sided sign-test p possible with 5 seeds: 0.0625.

**C vs B**

| seed | budget n | C best@n | B best@n | difference | higher |
|---|---|---|---|---|---|
| 0 | 278 | 0.4632 | 0.4203 | +0.0430 | C |
| 1 | 276 | 0.4745 | 0.5025 | -0.0280 | B |
| 2 | 280 | 0.5138 | 0.5040 | +0.0098 | C |
| 3 | 280 | 0.5434 | 0.4007 | +0.1427 | C |
| 4 | 275 | 0.4370 | 0.4873 | -0.0502 | B |

Mean difference +0.0234 (range -0.0502 to +0.1427). **NO DEMONSTRATED DIFFERENCE between C and B: C higher in 3, lower in 2, tied in 0 of 5 seeds.** Smallest two-sided sign-test p possible with 5 seeds: 0.0625.

### 9. What the LLM operators did inside arm C (from the per-call logs; accepted, i.e. valid, responses only)

| seed | mutate calls | positions changed | distinct position indices | most-used index (share) | distinct new letters (of 20) | most-used letter (share) | crossover calls | distinct cut values | cut values (count) |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 266 | 792 | 28 | 10 (12%) | 18 | G (21%) | 124 | 2 | 40: 107, 50: 17 |
| 1 | 266 | 804 | 26 | 3 (17%) | 19 | G (19%) | 124 | 2 | 40: 118, 50: 6 |
| 2 | 266 | 1032 | 41 | 10 (12%) | 19 | L (18%) | 117 | 2 | 40: 109, 50: 8 |
| 3 | 265 | 798 | 28 | 40 (12%) | 18 | G (19%) | 123 | 2 | 40: 104, 50: 19 |
| 4 | 266 | 882 | 33 | 10 (13%) | 19 | G (17%) | 122 | 2 | 40: 108, 50: 14 |

