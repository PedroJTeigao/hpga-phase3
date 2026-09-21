qwen2.5:7b, 100 crossover/segment calls per setting, genome length 63, bounds [30, 80], temperature 0.7, seed 0. Cuts are percentages; 99 possible values.

| setting | prose sentence says | worked example | fell back | requests/call | valid declarations | segments per call | distinct cuts (of 99) | modal cut (share of cuts) | cuts at 40 | cuts at 25 | declarations containing 40 / 25 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| current | 40 | 0-40:1, 40-100:2 present | 0/100 | 1.00 | 100 | {'2': 100} | 1 | 40 (100.0%) | 100 of 100 | 0 of 100 | 100 / 0 of 100 |
| absent | 40 | removed | 0/100 | 1.00 | 100 | {'2': 100} | 1 | 40 (100.0%) | 100 of 100 | 0 of 100 | 100 / 0 of 100 |
| prose25_example40 | 25 | 0-40:1, 40-100:2 present | 0/100 | 1.00 | 100 | {'2': 100} | 2 | 25 (97.0%) | 3 of 100 | 97 of 100 | 3 / 97 of 100 |
| absent_prose25 | 25 | removed | 0/100 | 1.01 | 100 | {'2': 89, '3': 10, '4': 1} | 3 | 25 (89.3%) | 0 of 112 | 100 of 112 | 0 / 100 of 100 |

Full cut distribution (value x count), most used first:

- **current**: 40x100
- **absent**: 40x100
- **prose25_example40**: 25x97, 40x3
- **absent_prose25**: 25x100, 75x11, 50x1

Prompt check (from the logged prompts, every attempt):

| setting | logged attempts | prose says 40 | prose says 25 | example 0-40 present | '40' outside Parent header lines | parent length 40 printed in a header (data) |
|---|---|---|---|---|---|---|
| current | 100 | 100 | 0 | 100 | 100 | 3 |
| absent | 100 | 100 | 0 | 0 | 100 | 3 |
| prose25_example40 | 100 | 0 | 100 | 100 | 100 | 3 |
| absent_prose25 | 101 | 0 | 101 | 0 | 0 | 5 |

Replication check against step 2 (same model, seed and prompt settings, run earlier):

- current: step 2 {'40': 100} vs this run {'40': 100} -> identical
- absent_prose25: step 2 {'25': 100, '50': 1, '75': 11} vs this run {'25': 100, '50': 1, '75': 11} -> identical
