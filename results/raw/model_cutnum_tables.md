100 crossover/segment calls per cell, genome length 63, bounds [30, 80], temperature 0.7, seed 0, worked example removed in every setting, prose sentence 'a boundary at N falls N% of the way ...' with N as shown. 99 possible cuts.

## qwen2.5:7b

| prose N | fell back | requests/call | valid declarations | segments per call | distinct cuts (of 99) | modal cut (share of cuts) | cuts = N (share of cuts) | calls containing N (of 100) | calls whose only cut is N (of 100) | cuts within +-5 of N | cuts in 45-55 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 10 | 0/100 | 1.01 | 100 | {'2': 94, '3': 5, '4': 1} | 13 | 30 (67.3%) | 1 of 107 (0.9%) | 1 | 0 | 0.9% | 3.7% |
| 25 | 0/100 | 1.01 | 100 | {'2': 89, '3': 10, '4': 1} | 3 | 25 (89.3%) | 100 of 112 (89.3%) | 100 | 89 | 89.3% | 0.9% |
| 37 | 0/100 | 1.00 | 100 | {'2': 100} | 1 | 37 (100.0%) | 100 of 100 (100.0%) | 100 | 100 | 100.0% | 0.0% |
| 60 | 0/100 | 1.00 | 100 | {'2': 100} | 2 | 60 (97.0%) | 97 of 100 (97.0%) | 97 | 97 | 97.0% | 0.0% |
| 90 | 0/100 | 1.04 | 100 | {'2': 80, '3': 20} | 9 | 30 (59.2%) | 31 of 120 (25.8%) | 31 | 11 | 25.8% | 5.0% |

Full cut distribution (value x count), most used first:

- **N=10**: 30x72, 40x9, 25x5, 20x4, 33x3, 35x3, 50x3, 34x2, 37x2, 10x1, 45x1, 60x1, 75x1
- **N=25**: 25x100, 75x11, 50x1
- **N=37**: 37x100
- **N=60**: 60x97, 40x3
- **N=90**: 30x71, 90x31, 45x6, 40x4, 35x3, 36x2, 34x1, 42x1, 43x1

First-attempt cuts (every call's first reply, valid or not), most used first:

- **N=10**: 30x72, 40x9, 25x5, 20x4, 33x3, 35x3, 50x3, 34x2, 10x1, 37x1, 38x1, 45x1, 60x1, 75x1
- **N=25**: 25x100, 75x10, 50x1
- **N=37**: 37x100
- **N=60**: 60x97, 40x3
- **N=90**: 30x70, 90x31, 45x6, 35x4, 40x3, 36x2, 43x2, 0x1, 34x1, 42x1

## gemma4:12b

| prose N | fell back | requests/call | valid declarations | segments per call | distinct cuts (of 99) | modal cut (share of cuts) | cuts = N (share of cuts) | calls containing N (of 100) | calls whose only cut is N (of 100) | cuts within +-5 of N | cuts in 45-55 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 10 | 0/100 | 1.00 | 100 | {'2': 100} | 2 | 50 (96.0%) | 0 of 100 (0.0%) | 0 | 0 | 0.0% | 96.0% |
| 25 | 0/100 | 1.00 | 100 | {'2': 100} | 1 | 50 (100.0%) | 0 of 100 (0.0%) | 0 | 0 | 0.0% | 100.0% |
| 37 | 0/100 | 1.00 | 100 | {'2': 100} | 2 | 50 (73.0%) | 27 of 100 (27.0%) | 27 | 27 | 27.0% | 73.0% |
| 60 | 0/100 | 1.00 | 100 | {'2': 100} | 2 | 60 (90.0%) | 90 of 100 (90.0%) | 90 | 90 | 90.0% | 10.0% |
| 90 | 0/100 | 1.00 | 100 | {'2': 100} | 1 | 50 (100.0%) | 0 of 100 (0.0%) | 0 | 0 | 0.0% | 100.0% |

Full cut distribution (value x count), most used first:

- **N=10**: 50x96, 40x4
- **N=25**: 50x100
- **N=37**: 50x73, 37x27
- **N=60**: 60x90, 50x10
- **N=90**: 50x100

First-attempt cuts (every call's first reply, valid or not), most used first:

- **N=10**: 50x96, 40x4
- **N=25**: 50x100
- **N=37**: 50x73, 37x27
- **N=60**: 60x90, 50x10
- **N=90**: 50x100

## Prompt check (from the logged prompts, every attempt, both models)

| model | prose N | logged attempts | prose sentence with N | 'e.g.' present | attempts with N anywhere but the 3 prose slots | parent length N printed in a header (data) |
|---|---|---|---|---|---|---|
| qwen2.5:7b | 10 | 101 | 101 | 0 | 0 | 0 |
| qwen2.5:7b | 25 | 101 | 101 | 0 | 0 | 0 |
| qwen2.5:7b | 37 | 100 | 100 | 0 | 0 | 4 |
| qwen2.5:7b | 60 | 100 | 100 | 0 | 0 | 4 |
| qwen2.5:7b | 90 | 104 | 104 | 0 | 0 | 0 |
| gemma4:12b | 10 | 100 | 100 | 0 | 0 | 0 |
| gemma4:12b | 25 | 100 | 100 | 0 | 0 | 0 |
| gemma4:12b | 37 | 100 | 100 | 0 | 0 | 4 |
| gemma4:12b | 60 | 100 | 100 | 0 | 0 | 4 |
| gemma4:12b | 90 | 100 | 100 | 0 | 0 | 0 |

## Replication against earlier runs (same model, seed, prompt)

- qwen2.5:7b N=25: step 2 `absent_prose25` {25: 100, 50: 1, 75: 11} vs this run {25: 100, 75: 11, 50: 1} -> identical
- gemma4:12b N=25: this run {50: 100} (no earlier file with this exact setting on disk; PHASE3_RESULTS.md 9.3 reports 50x100)
