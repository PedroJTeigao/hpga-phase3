100 crossover/segment calls per cell, genome length 63, bounds [30, 80], temperature 0.7, seed 0, worked example removed in every setting, prose sentence 'a boundary at N falls N% of the way ...' with N as shown. 99 possible cuts.

## qwen2.5:7b

| prose N | fell back | requests/call | valid declarations | segments per call | distinct cuts (of 99) | modal cut (share of cuts) | cuts = N (share of cuts) | calls containing N (of 100) | calls whose only cut is N (of 100) | cuts within +-5 of N | cuts in 45-55 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 33 | 0/100 | 1.00 | 100 | {'2': 100} | 1 | 33 (100.0%) | 100 of 100 (100.0%) | 100 | 100 | 100.0% | 0.0% |
| 45 | 0/100 | 1.00 | 100 | {'2': 100} | 1 | 45 (100.0%) | 100 of 100 (100.0%) | 100 | 100 | 100.0% | 100.0% |
| 70 | 0/100 | 1.00 | 100 | {'2': 99, '3': 1} | 7 | 70 (69.3%) | 70 of 101 (69.3%) | 70 | 69 | 69.3% | 0.0% |
| 80 | 0/100 | 1.02 | 100 | {'2': 96, '3': 4} | 4 | 80 (51.0%) | 53 of 104 (51.0%) | 53 | 49 | 51.0% | 0.0% |

Full cut distribution (value x count), most used first:

- **N=33**: 33x100
- **N=45**: 45x100
- **N=70**: 70x70, 35x22, 30x4, 34x2, 36x1, 42x1, 56x1
- **N=80**: 80x53, 40x49, 20x1, 25x1

First-attempt cuts (every call's first reply, valid or not), most used first:

- **N=33**: 33x100
- **N=45**: 45x100
- **N=70**: 70x70, 35x22, 30x4, 34x2, 36x1, 42x1, 56x1
- **N=80**: 80x53, 40x49, 20x1, 25x1

## gemma4:12b

| prose N | fell back | requests/call | valid declarations | segments per call | distinct cuts (of 99) | modal cut (share of cuts) | cuts = N (share of cuts) | calls containing N (of 100) | calls whose only cut is N (of 100) | cuts within +-5 of N | cuts in 45-55 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 33 | 0/100 | 1.00 | 100 | {'2': 99, '3': 1} | 3 | 33 (65.3%) | 66 of 101 (65.3%) | 66 | 65 | 65.3% | 33.7% |
| 45 | 0/100 | 1.00 | 100 | {'2': 100} | 2 | 50 (98.0%) | 2 of 100 (2.0%) | 2 | 2 | 100.0% | 100.0% |
| 70 | 0/100 | 1.00 | 100 | {'2': 100} | 2 | 70 (57.0%) | 57 of 100 (57.0%) | 57 | 57 | 57.0% | 43.0% |
| 80 | 0/100 | 1.00 | 100 | {'2': 100} | 1 | 50 (100.0%) | 0 of 100 (0.0%) | 0 | 0 | 0.0% | 100.0% |

Full cut distribution (value x count), most used first:

- **N=33**: 33x66, 50x34, 66x1
- **N=45**: 50x98, 45x2
- **N=70**: 70x57, 50x43
- **N=80**: 50x100

First-attempt cuts (every call's first reply, valid or not), most used first:

- **N=33**: 33x66, 50x34, 66x1
- **N=45**: 50x98, 45x2
- **N=70**: 70x57, 50x43
- **N=80**: 50x100

## Prompt check (from the logged prompts, every attempt, both models)

| model | prose N | logged attempts | prose sentence with N | 'e.g.' present | attempts with N anywhere but the 3 prose slots | parent length N printed in a header (data) |
|---|---|---|---|---|---|---|
| qwen2.5:7b | 33 | 100 | 100 | 0 | 0 | 2 |
| qwen2.5:7b | 45 | 100 | 100 | 0 | 0 | 3 |
| qwen2.5:7b | 70 | 100 | 100 | 0 | 0 | 2 |
| qwen2.5:7b | 80 | 102 | 102 | 0 | 0 | 5 |
| gemma4:12b | 33 | 100 | 100 | 0 | 0 | 2 |
| gemma4:12b | 45 | 100 | 100 | 0 | 0 | 3 |
| gemma4:12b | 70 | 100 | 100 | 0 | 0 | 2 |
| gemma4:12b | 80 | 100 | 100 | 0 | 0 | 1 |
