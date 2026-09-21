Per model, per setting: 100 crossover/segment calls, genome length 63, bounds [30, 80], temperature 0.7, seed 0. Cuts are percentages; 99 possible values (1-99). Each model is reported on its own.

## llama3.2:3b

| setting | fell back | requests/call | valid declarations | segments per call | distinct cut values (of 99) | modal cut (share of cuts) | cuts equal to 40 | declarations containing a 40 cut |
|---|---|---|---|---|---|---|---|---|
| a. current (prose number 40, example 0-40:1, 40-100:2) | 7/100 (0.070) | 1.63 | 93 | {'2': 7, '3': 62, '4': 18, '5': 5, '6': 1} | 7 | 40 (44.3%) | 93 of 210 (44.3%) | 93 of 93 |
| b. absent_prose25 (no example; prose number 25; no 40 in prompt or retry hint) | 4/100 (0.040) | 1.40 | 96 | {'3': 2, '4': 94} | 3 | 25/50 (33.6%) | 0 of 286 (0.0%) | 0 of 96 |

Full cut distribution (value x count), most used first:

- **current**: 40x93, 80x82, 60x19, 20x9, 50x3, 70x3, 75x1
- **absent_prose25**: 25x96, 50x96, 75x94

Prompt check (what the logged prompts actually contained, over every logged attempt):

| setting | logged attempts | prose says 40 | prose says 25 | example 0-40 present | any e.g. | '40' anywhere outside the Parent header lines | parent length 40 printed in a header (data) |
|---|---|---|---|---|---|---|---|
| current | 163 | 163 | 0 | 163 | 163 | 163 | 10 |
| absent_prose25 | 140 | 0 | 140 | 0 | 0 | 0 | 3 |

For reference, step 1 (same prompt as setting a, different RNG stream because mutate ran first): fallback 2/100, 7 distinct cuts, modal 40 (46.9%).

## qwen2.5:7b

| setting | fell back | requests/call | valid declarations | segments per call | distinct cut values (of 99) | modal cut (share of cuts) | cuts equal to 40 | declarations containing a 40 cut |
|---|---|---|---|---|---|---|---|---|
| a. current (prose number 40, example 0-40:1, 40-100:2) | 0/100 (0.000) | 1.00 | 100 | {'2': 100} | 1 | 40 (100.0%) | 100 of 100 (100.0%) | 100 of 100 |
| b. absent_prose25 (no example; prose number 25; no 40 in prompt or retry hint) | 0/100 (0.000) | 1.01 | 100 | {'2': 89, '3': 10, '4': 1} | 3 | 25 (89.3%) | 0 of 112 (0.0%) | 0 of 100 |

Full cut distribution (value x count), most used first:

- **current**: 40x100
- **absent_prose25**: 25x100, 75x11, 50x1

Prompt check (what the logged prompts actually contained, over every logged attempt):

| setting | logged attempts | prose says 40 | prose says 25 | example 0-40 present | any e.g. | '40' anywhere outside the Parent header lines | parent length 40 printed in a header (data) |
|---|---|---|---|---|---|---|---|
| current | 100 | 100 | 0 | 100 | 100 | 100 | 3 |
| absent_prose25 | 101 | 0 | 101 | 0 | 0 | 0 | 5 |

For reference, step 1 (same prompt as setting a, different RNG stream because mutate ran first): fallback 0/100, 1 distinct cuts, modal 40 (100.0%).

## mistral:7b

| setting | fell back | requests/call | valid declarations | segments per call | distinct cut values (of 99) | modal cut (share of cuts) | cuts equal to 40 | declarations containing a 40 cut |
|---|---|---|---|---|---|---|---|---|
| a. current (prose number 40, example 0-40:1, 40-100:2) | 4/100 (0.040) | 1.18 | 96 | {'3': 95, '4': 1} | 11 | 40 (49.2%) | 95 of 193 (49.2%) | 95 of 96 |
| b. absent_prose25 (no example; prose number 25; no 40 in prompt or retry hint) | 47/100 (0.470) | 2.24 | 53 | {'3': 1, '4': 51, '5': 1} | 13 | 25 (33.3%) | 0 of 159 (0.0%) | 0 of 53 |

Full cut distribution (value x count), most used first:

- **current**: 40x95, 80x88, 81x2, 30x1, 60x1, 76x1, 78x1, 79x1, 82x1, 85x1, 86x1
- **absent_prose25**: 25x53, 50x40, 75x28, 80x22, 45x4, 55x3, 49x2, 70x2, 48x1, 52x1, 53x1, 60x1, 65x1

Prompt check (what the logged prompts actually contained, over every logged attempt):

| setting | logged attempts | prose says 40 | prose says 25 | example 0-40 present | any e.g. | '40' anywhere outside the Parent header lines | parent length 40 printed in a header (data) |
|---|---|---|---|---|---|---|---|
| current | 118 | 118 | 0 | 118 | 118 | 118 | 10 |
| absent_prose25 | 224 | 0 | 224 | 0 | 0 | 0 | 8 |

For reference, step 1 (same prompt as setting a, different RNG stream because mutate ran first): fallback 13/100, 5 distinct cuts, modal 40 (49.4%).

