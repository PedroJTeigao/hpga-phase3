Offline: is the chosen position predicted by the letter that sits there? (no model calls; calls that fell back are excluded)

| source | calls used | letters: chi2 (df 19) vs A (uniform position), MC p | letters: chi2 vs B (positions kept, content shuffled), MC p | classes: chi2 (df 3) vs A, p | classes: chi2 vs B, p |
|---|---|---|---|---|---|
| gemma4:12b (step 1, n=100) | 100 | 10.9, p=0.923 | 12.1, p=0.740 | 1.07, p=0.787 | 0.39, p=0.924 |
| gemma4:12b (P3 sec. 9 run, n=400) | 400 | 29.0, p=0.058 | 25.7, p=0.035 | 2.21, p=0.524 | 1.49, p=0.600 |
| llama3.2:3b (step 1) | 59 | 22.8, p=0.229 | 20.9, p=0.238 | 1.12, p=0.769 | 1.79, p=0.591 |
| qwen2.5:7b (step 1) | 100 | 19.0, p=0.433 | 9.9, p=0.718 | 4.15, p=0.235 | 2.13, p=0.335 |
| mistral:7b (step 1) | 96 | 10.9, p=0.922 | 10.8, p=0.920 | 1.59, p=0.659 | 1.03, p=0.795 |

Monte Carlo p-values, 20000 draws each (floor 5.0e-05). Null A = position uniform over the genome; null B = the model's own positions kept but paired with genomes from other calls. Expected count per letter is ~5 at n=100 and ~20 at n=400, so a modest content preference would not be detectable in the n=100 rows.

### Residue classes at the chosen position

| source | class | share of all genome letters | share at chosen positions | expected share under B |
|---|---|---|---|---|
| gemma4:12b (step 1, n=100) | hydrophobic | 35.2% | 36.0% | 36.0% |
| gemma4:12b (step 1, n=100) | charged | 24.7% | 23.0% | 25.3% |
| gemma4:12b (step 1, n=100) | polar | 30.0% | 28.0% | 27.2% |
| gemma4:12b (step 1, n=100) | special | 10.1% | 13.0% | 11.6% |
| gemma4:12b (P3 sec. 9 run, n=400) | hydrophobic | 35.2% | 36.8% | 35.0% |
| gemma4:12b (P3 sec. 9 run, n=400) | charged | 24.8% | 24.2% | 25.5% |
| gemma4:12b (P3 sec. 9 run, n=400) | polar | 29.9% | 27.2% | 28.9% |
| gemma4:12b (P3 sec. 9 run, n=400) | special | 10.2% | 11.8% | 10.6% |
| llama3.2:3b (step 1) | hydrophobic | 35.7% | 35.6% | 35.6% |
| llama3.2:3b (step 1) | charged | 24.5% | 23.7% | 23.9% |
| llama3.2:3b (step 1) | polar | 30.1% | 27.1% | 31.6% |
| llama3.2:3b (step 1) | special | 9.7% | 13.6% | 8.9% |
| qwen2.5:7b (step 1) | hydrophobic | 35.2% | 42.0% | 36.3% |
| qwen2.5:7b (step 1) | charged | 24.7% | 26.0% | 25.0% |
| qwen2.5:7b (step 1) | polar | 30.0% | 21.0% | 25.9% |
| qwen2.5:7b (step 1) | special | 10.1% | 11.0% | 12.9% |
| mistral:7b (step 1) | hydrophobic | 35.0% | 34.4% | 34.3% |
| mistral:7b (step 1) | charged | 25.0% | 30.2% | 26.1% |
| mistral:7b (step 1) | polar | 30.0% | 26.0% | 29.2% |
| mistral:7b (step 1) | special | 10.0% | 9.4% | 10.4% |

### Letter-level detail (share of all genome letters / share at chosen positions / expected under B)

**gemma4:12b (step 1, n=100)**, 100 calls:

| letter | A | C | D | E | F | G | H | I | K | L | M | N | P | Q | R | S | T | V | W | Y |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| available (%) | 5.2 | 4.9 | 4.7 | 5.2 | 5.4 | 4.8 | 5.0 | 4.7 | 5.0 | 5.0 | 5.2 | 5.1 | 5.3 | 5.2 | 4.8 | 4.9 | 4.7 | 4.8 | 4.8 | 5.3 |
| at chosen position (count) | 6 | 4 | 5 | 4 | 5 | 8 | 3 | 4 | 7 | 4 | 3 | 5 | 5 | 4 | 4 | 3 | 5 | 9 | 5 | 7 |
| expected under B (count) | 3.6 | 3.5 | 3.9 | 5.2 | 5.7 | 6.8 | 6.1 | 4.9 | 4.3 | 5.4 | 4.8 | 4.7 | 4.9 | 6.2 | 5.6 | 4.0 | 3.9 | 5.5 | 6.1 | 5.0 |

**gemma4:12b (P3 sec. 9 run, n=400)**, 400 calls:

| letter | A | C | D | E | F | G | H | I | K | L | M | N | P | Q | R | S | T | V | W | Y |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| available (%) | 5.1 | 4.9 | 4.8 | 5.1 | 4.9 | 5.1 | 5.1 | 5.0 | 5.0 | 4.9 | 5.1 | 5.0 | 5.1 | 5.0 | 4.6 | 4.9 | 4.9 | 4.9 | 5.2 | 5.2 |
| at chosen position (count) | 23 | 10 | 27 | 14 | 21 | 25 | 20 | 15 | 22 | 16 | 16 | 26 | 22 | 22 | 14 | 22 | 10 | 28 | 28 | 19 |
| expected under B (count) | 19.7 | 17.5 | 22.0 | 19.7 | 21.3 | 21.6 | 22.5 | 19.7 | 20.4 | 20.1 | 18.6 | 17.2 | 20.8 | 24.9 | 17.3 | 18.2 | 18.5 | 20.4 | 20.4 | 19.3 |

**llama3.2:3b (step 1)**, 59 calls:

| letter | A | C | D | E | F | G | H | I | K | L | M | N | P | Q | R | S | T | V | W | Y |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| available (%) | 5.2 | 5.5 | 4.3 | 5.1 | 5.6 | 4.5 | 5.3 | 5.1 | 5.3 | 5.0 | 5.6 | 5.0 | 5.1 | 5.4 | 4.6 | 5.0 | 4.5 | 5.0 | 4.0 | 4.8 |
| at chosen position (count) | 2 | 6 | 5 | 1 | 2 | 4 | 7 | 2 | 0 | 5 | 4 | 1 | 4 | 4 | 1 | 3 | 1 | 3 | 3 | 1 |
| expected under B (count) | 3.7 | 4.1 | 2.6 | 2.9 | 3.2 | 2.3 | 3.3 | 2.7 | 3.2 | 2.8 | 3.6 | 3.0 | 2.9 | 3.7 | 2.1 | 3.1 | 2.9 | 3.1 | 1.9 | 1.9 |

**qwen2.5:7b (step 1)**, 100 calls:

| letter | A | C | D | E | F | G | H | I | K | L | M | N | P | Q | R | S | T | V | W | Y |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| available (%) | 5.2 | 4.9 | 4.7 | 5.2 | 5.4 | 4.8 | 5.0 | 4.7 | 5.0 | 5.0 | 5.2 | 5.1 | 5.3 | 5.2 | 4.8 | 4.9 | 4.7 | 4.8 | 4.8 | 5.3 |
| at chosen position (count) | 2 | 4 | 3 | 5 | 6 | 7 | 8 | 9 | 8 | 6 | 6 | 2 | 4 | 4 | 2 | 4 | 4 | 5 | 8 | 3 |
| expected under B (count) | 2.4 | 3.2 | 3.2 | 5.2 | 5.5 | 7.3 | 6.8 | 5.7 | 5.0 | 5.9 | 4.6 | 4.3 | 5.6 | 6.5 | 4.8 | 3.6 | 4.3 | 5.8 | 6.4 | 3.9 |

**mistral:7b (step 1)**, 96 calls:

| letter | A | C | D | E | F | G | H | I | K | L | M | N | P | Q | R | S | T | V | W | Y |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| available (%) | 5.2 | 5.1 | 4.9 | 5.2 | 5.3 | 4.8 | 5.2 | 4.9 | 5.0 | 4.8 | 5.3 | 5.0 | 5.2 | 5.0 | 4.8 | 4.9 | 4.6 | 4.8 | 4.7 | 5.5 |
| at chosen position (count) | 5 | 5 | 6 | 7 | 5 | 2 | 5 | 4 | 5 | 3 | 4 | 3 | 7 | 4 | 6 | 5 | 4 | 3 | 9 | 4 |
| expected under B (count) | 5.0 | 4.9 | 4.7 | 5.6 | 4.9 | 5.0 | 4.7 | 5.2 | 5.1 | 4.1 | 5.0 | 4.9 | 4.9 | 4.4 | 5.0 | 4.4 | 4.2 | 4.4 | 4.4 | 5.2 |

