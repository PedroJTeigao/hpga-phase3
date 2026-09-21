100 mutate/position calls per cell, temperature 0.7, seed 0, one change per call (k = 1). Each model reported on its own.

## gemma4:12b

| genome length N | fell back | requests/call | successful calls | distinct positions (of N; expected if uniform) | modal position (share) | modal position / (N-1) | top-3 share | chi2 vs uniform (df N-1), MC p | choices at multiples of 10 (share; uniform would give) |
|---|---|---|---|---|---|---|---|---|---|
| 40 | 0/100 | 1.01 | 100 | 13 (36.8) | 10 (38.0%) | 0.256 | 70.0% | 785, p=5e-05 | 39.0% (10.0%) |
| 63 | 0/100 | 1.00 | 100 | 17 (50.3) | 10 (38.0%) | 0.161 | 62.0% | 1083, p=5e-05 | 41.0% (11.1%) |
| 80 | 0/100 | 1.00 | 100 | 13 (57.3) | 10 (42.0%) | 0.127 | 64.0% | 1662, p=5e-05 | 44.0% (10.0%) |

Full distribution (position x count), most used first:

- **N=40**: 10x38, 3x24, 2x8, 12x7, 15x6, 1x4, 13x3, 4x2, 11x2, 14x2, 18x2, 0x1, 5x1
- **N=63**: 10x38, 3x12, 12x12, 14x6, 11x5, 15x5, 4x4, 5x4, 2x3, 21x3, 22x2, 0x1, 1x1, 13x1, 20x1, 30x1, 32x1
- **N=80**: 10x42, 3x11, 12x11, 15x9, 7x7, 4x5, 14x4, 2x3, 11x3, 20x2, 5x1, 31x1, 34x1

Predictions from the N=63 mode (10): absolute hypothesis -> the same index at N=40 and N=80 (10, 10); relative hypothesis -> 6.3 at N=40 and 12.7 at N=80. Observed modal positions: N=40: 10, N=63: 10, N=80: 10.

Replication at N=63: step 1 position counts identical to this run.

## qwen2.5:7b

| genome length N | fell back | requests/call | successful calls | distinct positions (of N; expected if uniform) | modal position (share) | modal position / (N-1) | top-3 share | chi2 vs uniform (df N-1), MC p | choices at multiples of 10 (share; uniform would give) |
|---|---|---|---|---|---|---|---|---|---|
| 40 | 0/100 | 1.25 | 100 | 4 (36.8) | 0 (54.0%) | 0.000 | 99.0% | 1506, p=5e-05 | 83.0% (10.0%) |
| 63 | 0/100 | 1.03 | 100 | 6 (50.3) | 10 (49.0%) | 0.161 | 83.0% | 1926, p=5e-05 | 74.0% (11.1%) |
| 80 | 0/100 | 1.01 | 100 | 5 (57.3) | 10 (67.0%) | 0.127 | 97.0% | 4126, p=5e-05 | 97.0% (10.0%) |

Full distribution (position x count), most used first:

- **N=40**: 0x54, 10x29, 1x16, 4x1
- **N=63**: 10x49, 0x25, 2x9, 1x8, 3x6, 12x3
- **N=80**: 10x67, 0x28, 4x2, 20x2, 3x1

Predictions from the N=63 mode (10): absolute hypothesis -> the same index at N=40 and N=80 (10, 10); relative hypothesis -> 6.3 at N=40 and 12.7 at N=80. Observed modal positions: N=40: 0, N=63: 10, N=80: 10.

Replication at N=63: step 1 position counts DIFFERENT from this run.

