
## dlogit: Δlogit at changed positions (substitution arms only) vs realised fitness gain

| subset | n | Pearson r | p~ | Spearman rho | p~ |
|---|---|---|---|---|---|
| **pooled** | 3192 | +0.025 | 0.16 | +0.013 | 0.45 |
| S1 | 798 | -0.044 | 0.22 | -0.029 | 0.42 |
| S2 | 798 | +0.044 | 0.21 | +0.036 | 0.31 |
| S3 | 798 | -0.012 | 0.73 | -0.015 | 0.66 |
| B | 798 | +0.089 | 0.012 | +0.063 | 0.073 |
| pooled, improved only | 994 | +0.027 | 0.39 | +0.017 | 0.6 |
| pooled, not improved | 2198 | +0.036 | 0.088 | +0.011 | 0.61 |

- sign agreement: 1567/3159 = 49.6%
- moves the model prefers (Δ>0): 1626, improved 514 (31.6%)
- moves the model dislikes (Δ<0): 1533, improved 480 (31.3%)

## dPLL: ΔPLL = PLL(child) − PLL(base) (all arms, indels included) vs realised fitness gain

| subset | n | Pearson r | p~ | Spearman rho | p~ |
|---|---|---|---|---|---|
| **pooled** | 3990 | -0.020 | 0.2 | -0.038 | 0.018 |
| S1 | 798 | -0.082 | 0.02 | -0.074 | 0.037 |
| S2 | 798 | +0.101 | 0.0042 | +0.064 | 0.069 |
| S3 | 798 | -0.049 | 0.16 | -0.069 | 0.05 |
| S4 | 798 | -0.087 | 0.014 | -0.091 | 0.0097 |
| B | 798 | +0.098 | 0.0057 | +0.022 | 0.53 |
| pooled, improved only | 1298 | +0.032 | 0.25 | +0.027 | 0.32 |
| pooled, not improved | 2692 | -0.003 | 0.87 | -0.032 | 0.095 |

- sign agreement: 1938/3957 = 49.0%
- moves the model prefers (Δ>0): 1981, improved 630 (31.8%)
- moves the model dislikes (Δ<0): 1976, improved 668 (33.8%)

## Base genome: mean per-position log-likelihood vs its own TM fitness

| subset | n | Pearson r | p~ | Spearman rho | p~ |
|---|---|---|---|---|---|
| **pooled unique (arm,seed,base)** | 3197 | +0.146 | 9.9e-17 | +0.149 | 2.2e-17 |
| S1 | 562 | +0.218 | 1.6e-07 | +0.146 | 0.00051 |
| S2 | 680 | +0.129 | 0.00073 | +0.050 | 0.19 |
| S3 | 617 | -0.101 | 0.012 | -0.078 | 0.053 |
| S4 | 697 | +0.146 | 0.00011 | +0.236 | 2.2e-10 |
| B | 641 | +0.194 | 7e-07 | +0.151 | 0.00012 |
| seed 0 (all arms) | 1063 | +0.058 | 0.061 | +0.036 | 0.24 |
| seed 1 (all arms) | 1081 | +0.160 | 1.2e-07 | +0.108 | 0.00038 |
| seed 2 (all arms) | 1053 | +0.087 | 0.0049 | +0.164 | 8.3e-08 |

### The breeding-step confound

Bases come from an evolving population, so both quantities drift with step:

| quantity vs breeding step | n | r | p~ |
|---|---|---|---|
| mean per-position LL | 3197 | +0.118 | 1.8e-11 |
| base TM fitness | 3197 | +0.460 | 6e-174 |

Conditioning on step (thirds as in MOVE_CLASS.md §4, then single steps):

| LL vs fitness, within band | n | r | p~ |
|---|---|---|---|
| early 0–5 | 1081 | +0.177 | 3.9e-09 |
| middle 6–11 | 989 | +0.004 | 0.89 |
| late 12–18 | 1127 | +0.173 | 4.8e-09 |
| single step 0 | 203 | +0.193 | 0.0057 |
| single step 9 | 163 | +0.112 | 0.15 |
| single step 18 | 176 | +0.220 | 0.0032 |

### Calibration against the native sequence

- native 7UR7 chain A (the fitness reference): mean per-position LL **-2.253**
- the 3197 unique GA bases: mean -3.111, min -3.320, max -2.285
- their TM fitness: mean 0.3322, min 0.1585, max 0.6522
- bases scoring above the native sequence: 0 of 3197
