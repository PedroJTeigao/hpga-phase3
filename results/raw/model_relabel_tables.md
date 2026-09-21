100 mutate/position calls per cell, genome length 63, temperature 0.7, seed 0, one change per call. Labels are the numbers the model is shown and answers with; the genome is identical in every setting.

## gemma4:12b

| setting (labels shown) | fell back | requests/call | successful calls | distinct accepted labels (of 63) | modal accepted label (share) | modal label as index (label - offset) | modal label is a multiple of 10 | first replies outside the stated range | modal first-reply label (share of 100) |
|---|---|---|---|---|---|---|---|---|---|
| 0-62 (as shipped) | 0/100 | 1.00 | 100 | 17 | 10 (38.0%) | 10 | yes | 0 of 100 | 10 (38%) |
| 100-162 | 0/100 | 1.01 | 100 | 9 | 101 (25.0%) | 1 | no | 0 of 100 | 101 (26%) |
| 107-169 (extra setting) | 0/100 | 1.04 | 100 | 14 | 107 (20.0%) | 0 | no | 0 of 100 | 110 (21%) |

**0-62 (as shipped)**
- accepted labels (label x count): 10x38, 3x12, 12x12, 14x6, 11x5, 15x5, 4x4, 5x4, 2x3, 21x3, 22x2, 0x1, 1x1, 13x1, 20x1, 30x1, 32x1
- first-reply labels (label x count): 10x38, 3x12, 12x12, 14x6, 11x5, 15x5, 4x4, 5x4, 2x3, 21x3, 22x2, 0x1, 1x1, 13x1, 20x1, 30x1, 32x1
- first replies outside the stated range: none

**100-162**
- accepted labels (label x count): 101x25, 104x18, 100x16, 103x16, 105x12, 110x8, 102x3, 106x1, 130x1
- first-reply labels (label x count): 101x26, 104x18, 103x16, 100x15, 105x12, 110x8, 102x3, 106x1, 130x1
- first replies outside the stated range: none

**107-169 (extra setting)**
- accepted labels (label x count): 107x20, 110x20, 111x16, 108x10, 112x8, 115x7, 114x6, 109x3, 118x3, 113x2, 123x2, 120x1, 121x1, 140x1
- first-reply labels (label x count): 110x21, 107x16, 111x16, 108x11, 112x8, 114x7, 115x7, 109x3, 113x3, 118x3, 123x2, 120x1, 121x1, 140x1
- first replies outside the stated range: none

## qwen2.5:7b

| setting (labels shown) | fell back | requests/call | successful calls | distinct accepted labels (of 63) | modal accepted label (share) | modal label as index (label - offset) | modal label is a multiple of 10 | first replies outside the stated range | modal first-reply label (share of 100) |
|---|---|---|---|---|---|---|---|---|---|
| 0-62 (as shipped) | 0/100 | 1.00 | 100 | 6 | 10 (55.0%) | 10 | yes | 0 of 100 | 10 (55%) |
| 100-162 | 0/100 | 1.00 | 100 | 2 | 100 (96.0%) | 0 | yes | 0 of 100 | 100 (96%) |
| 107-169 (extra setting) | 0/100 | 1.00 | 100 | 2 | 108 (99.0%) | 1 | no | 0 of 100 | 108 (99%) |

**0-62 (as shipped)**
- accepted labels (label x count): 10x55, 2x12, 0x11, 3x11, 1x9, 12x2
- first-reply labels (label x count): 10x55, 2x12, 0x11, 3x11, 1x9, 12x2
- first replies outside the stated range: none

**100-162**
- accepted labels (label x count): 100x96, 101x4
- first-reply labels (label x count): 100x96, 101x4
- first replies outside the stated range: none

**107-169 (extra setting)**
- accepted labels (label x count): 108x99, 110x1
- first-reply labels (label x count): 108x99, 110x1
- first replies outside the stated range: none

