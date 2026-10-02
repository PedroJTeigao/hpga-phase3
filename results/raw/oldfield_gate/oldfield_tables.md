# OLD-slot gate: generated tables (experiments/probe_mutate_oldfield.py summarize)

Repeated-letter rate = parsed lines whose NEW equals the letter actually at that position (OLD ignored), the same definition under both formats. Invalid = strict validity of the format used.

## First attempts

| model | condition | format | calls | attempts | invalid attempts | fallbacks | lines | repeated-letter lines (rate) | OLD accuracy | OLD correct and NEW = OLD |
|---|---|---|---|---|---|---|---|---|---|---|
| gemma4:12b | C1 | old | 100 | 100 | 0 | 0 | 100 | 0 (0.000) |  |  |
| gemma4:12b | C1 | new | 100 | 100 | 28 | 2 | 100 | 0 (0.000) | 0.720 | 0 |
| gemma4:12b | C2 | old | 200 | 200 | 5 | 1 | 600 | 5 (0.008) |  |  |
| gemma4:12b | C2 | new | 200 | 200 | 175 | 126 | 600 | 5 (0.008) | 0.617 | 0 |
| gemma4:12b | C3 | old | 200 | 200 | 3 | 0 | 640 | 3 (0.005) |  |  |
| gemma4:12b | C3 | new | 200 | 200 | 186 | 136 | 640 | 8 (0.013) | 0.580 | 0 |
| qwen2.5:7b | C1 | old | 100 | 100 | 0 | 0 | 100 | 0 (0.000) |  |  |
| qwen2.5:7b | C1 | new | 100 | 100 | 82 | 13 | 174 | 9 (0.052) | 0.529 | 0 |
| qwen2.5:7b | C2 | old | 200 | 200 | 12 | 2 | 600 | 14 (0.023) |  |  |
| qwen2.5:7b | C2 | new | 200 | 200 | 199 | 199 | 600 | 34 (0.057) | 0.160 | 0 |
| qwen2.5:7b | C3 | old | 200 | 200 | 23 | 5 | 640 | 26 (0.041) |  |  |
| qwen2.5:7b | C3 | new | 200 | 200 | 200 | 199 | 640 | 35 (0.055) | 0.127 | 0 |
| mistral:7b | C1 | old | 100 | 100 | 30 | 4 | 126 | 7 (0.056) |  |  |
| mistral:7b | C1 | new | 100 | 100 | 95 | 85 | 165 | 5 (0.030) | 0.085 | 0 |
| mistral:7b | C2 | old | 200 | 200 | 200 | 200 | 400 | 17 (0.043) |  |  |
| mistral:7b | C2 | new | 200 | 200 | 199 | 199 | 600 | 18 (0.030) | 0.082 | 0 |
| mistral:7b | C3 | old | 200 | 200 | 199 | 199 | 441 | 24 (0.054) |  |  |
| mistral:7b | C3 | new | 200 | 200 | 200 | 199 | 640 | 29 (0.045) | 0.083 | 0 |
| llama3.2:3b | C1 | old | 100 | 100 | 100 | 41 | 200 | 4 (0.020) |  |  |
| llama3.2:3b | C1 | new | 100 | 100 | 96 | 86 | 300 | 3 (0.010) | 0.257 | 0 |
| llama3.2:3b | C2 | old | 200 | 200 | 40 | 5 | 600 | 24 (0.040) |  |  |
| llama3.2:3b | C2 | new | 200 | 200 | 200 | 199 | 600 | 26 (0.043) | 0.092 | 0 |
| llama3.2:3b | C3 | old | 200 | 200 | 47 | 11 | 640 | 32 (0.050) |  |  |
| llama3.2:3b | C3 | new | 200 | 200 | 200 | 200 | 640 | 27 (0.042) | 0.077 | 0 |

## All attempts

| model | condition | format | calls | attempts | invalid attempts | fallbacks | lines | repeated-letter lines (rate) | OLD accuracy | OLD correct and NEW = OLD |
|---|---|---|---|---|---|---|---|---|---|---|
| gemma4:12b | C1 | old | 100 | 100 | 0 | 0 | 100 | 0 (0.000) |  |  |
| gemma4:12b | C1 | new | 100 | 134 | 36 | 2 | 134 | 1 (0.007) | 0.731 | 0 |
| gemma4:12b | C2 | old | 200 | 207 | 8 | 1 | 621 | 8 (0.013) |  |  |
| gemma4:12b | C2 | new | 200 | 520 | 446 | 126 | 1560 | 16 (0.010) | 0.633 | 0 |
| gemma4:12b | C3 | old | 200 | 203 | 3 | 0 | 650 | 3 (0.005) |  |  |
| gemma4:12b | C3 | new | 200 | 548 | 484 | 136 | 1763 | 23 (0.013) | 0.618 | 0 |
| qwen2.5:7b | C1 | old | 100 | 100 | 0 | 0 | 100 | 0 (0.000) |  |  |
| qwen2.5:7b | C1 | new | 100 | 208 | 121 | 13 | 282 | 12 (0.043) | 0.571 | 0 |
| qwen2.5:7b | C2 | old | 200 | 216 | 18 | 2 | 648 | 20 (0.031) |  |  |
| qwen2.5:7b | C2 | new | 200 | 598 | 597 | 199 | 1794 | 93 (0.052) | 0.164 | 0 |
| qwen2.5:7b | C3 | old | 200 | 230 | 35 | 5 | 736 | 39 (0.053) |  |  |
| qwen2.5:7b | C3 | new | 200 | 600 | 599 | 199 | 1920 | 100 (0.052) | 0.129 | 0 |
| mistral:7b | C1 | old | 100 | 138 | 42 | 4 | 176 | 7 (0.040) |  |  |
| mistral:7b | C1 | new | 100 | 284 | 269 | 85 | 393 | 13 (0.033) | 0.076 | 0 |
| mistral:7b | C2 | old | 200 | 600 | 600 | 200 | 1200 | 65 (0.054) |  |  |
| mistral:7b | C2 | new | 200 | 598 | 597 | 199 | 1794 | 71 (0.040) | 0.081 | 0 |
| mistral:7b | C3 | old | 200 | 598 | 597 | 199 | 1319 | 68 (0.052) |  |  |
| mistral:7b | C3 | new | 200 | 599 | 598 | 199 | 1917 | 92 (0.048) | 0.086 | 0 |
| llama3.2:3b | C1 | old | 100 | 266 | 207 | 41 | 479 | 8 (0.017) |  |  |
| llama3.2:3b | C1 | new | 100 | 286 | 272 | 86 | 798 | 11 (0.014) | 0.259 | 0 |
| llama3.2:3b | C2 | old | 200 | 254 | 59 | 5 | 762 | 30 (0.039) |  |  |
| llama3.2:3b | C2 | new | 200 | 599 | 598 | 199 | 1797 | 68 (0.038) | 0.100 | 0 |
| llama3.2:3b | C3 | old | 200 | 265 | 76 | 11 | 865 | 40 (0.046) |  |  |
| llama3.2:3b | C3 | new | 200 | 600 | 600 | 200 | 1920 | 76 (0.040) | 0.093 | 0 |

## Invalid attempts by kind (all attempts)

| model | condition | format | kinds |
|---|---|---|---|
| gemma4:12b | C1 | old | {} |
| gemma4:12b | C1 | new | {'OLD wrong (NEW differs)': 35, 'NEW equals the current letter': 1} |
| gemma4:12b | C2 | old | {'NEW equals the current letter': 8} |
| gemma4:12b | C2 | new | {'OLD wrong (NEW differs)': 430, 'NEW equals the current letter': 16} |
| gemma4:12b | C3 | old | {'NEW equals the current letter': 3} |
| gemma4:12b | C3 | new | {'OLD wrong (NEW differs)': 461, 'NEW equals the current letter': 23} |
| qwen2.5:7b | C1 | old | {} |
| qwen2.5:7b | C1 | new | {'OLD wrong (NEW differs)': 68, 'wrong number of lines': 43, 'NEW equals the current letter': 10} |
| qwen2.5:7b | C2 | old | {'NEW equals the current letter': 18} |
| qwen2.5:7b | C2 | new | {'OLD wrong (NEW differs)': 508, 'NEW equals the current letter': 89} |
| qwen2.5:7b | C3 | old | {'NEW equals the current letter': 35} |
| qwen2.5:7b | C3 | new | {'OLD wrong (NEW differs)': 506, 'NEW equals the current letter': 93} |
| mistral:7b | C1 | old | {'wrong number of lines': 38, 'NEW equals the current letter': 4} |
| mistral:7b | C1 | new | {'wrong number of lines': 109, 'OLD wrong (NEW differs)': 158, 'NEW equals the current letter': 2} |
| mistral:7b | C2 | old | {'wrong number of lines': 600} |
| mistral:7b | C2 | new | {'OLD wrong (NEW differs)': 532, 'NEW equals the current letter': 65} |
| mistral:7b | C3 | old | {'wrong number of lines': 597} |
| mistral:7b | C3 | new | {'OLD wrong (NEW differs)': 511, 'NEW equals the current letter': 87} |
| llama3.2:3b | C1 | old | {'wrong number of lines': 206, 'NEW equals the current letter': 1} |
| llama3.2:3b | C1 | new | {'wrong number of lines': 249, 'other': 1, 'OLD wrong (NEW differs)': 21, 'NEW equals the current letter': 1} |
| llama3.2:3b | C2 | old | {'other': 32, 'NEW equals the current letter': 27} |
| llama3.2:3b | C2 | new | {'OLD wrong (NEW differs)': 532, 'NEW equals the current letter': 65, 'other': 1} |
| llama3.2:3b | C3 | old | {'other': 39, 'NEW equals the current letter': 37} |
| llama3.2:3b | C3 | new | {'OLD wrong (NEW differs)': 527, 'NEW equals the current letter': 73} |

## C1 old format against MODEL_HETEROGENEITY_STEP1's logged mutate prompts

- gemma4:12b: 100 of 100 prompts identical (logged attempts 100 here, 100 in step 1); first mismatch at None
- qwen2.5:7b: 100 of 100 prompts identical (logged attempts 100 here, 100 in step 1); first mismatch at None
- mistral:7b: 138 of 138 prompts identical (logged attempts 138 here, 138 in step 1); first mismatch at None
- llama3.2:3b: 266 of 266 prompts identical (logged attempts 266 here, 266 in step 1); first mismatch at None
