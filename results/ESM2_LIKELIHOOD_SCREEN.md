# Does a protein language model's preference predict fitness here?

**A screen, not an arm.** No GA was run and no LLM or ESM-2 proposed any move. This scores moves that the deterministic move classes already made (`results/MOVE_CLASS.md`) with ESM-2 150M's masked likelihood, and asks whether the model's preference tracks the realised TM-score change. **It does not.** Read the limits section before carrying this anywhere: a flat correlation over the bulk of a distribution does not establish what the model's *top-ranked* proposal would do.

p-values throughout are a **two-sided normal approximation on Fisher's z**, computed in `experiments/analyse_esm2_likelihood.py` because this venv has no scipy. They are **uncorrected for multiple comparisons**: sections 1 and 2 report **20 tests** (5 arms × 2 metrics × 2 coefficients), so a scattering of nominal p < 0.05 is expected under the null. Sign consistency across arms is the thing to read, not any individual p.

## The question, and why it is asked this way

`results/OPERATOR_DOES_NOT_MATTER.md` concluded that on this problem, at this budget, the operator is not what determines search quality; `results/PROJECT_SUMMARY.md` §3.7 then showed that varying the deterministic move class changes the per-move hit rate but not final fitness. The next arm under consideration was ESM-2 as the operator: instead of an LLM choosing edits from prose, use a protein language model's own likelihoods to propose mutations.

That arm rests on an assumption that can be tested **before building anything**. The move-class bases files already hold, for 3,990 moves, the base genome, the child genome and both fitnesses. If ESM-2's preference carries information about which moves help, it should show up there. The moves were produced by S1–S4 and B for reasons wholly unrelated to ESM-2, so they are an unbiased sample with respect to the model — a stricter test than scoring moves the model itself chose.

## What was computed

Model: `facebook/esm2_t30_150M_UR50D` (30 layers, hidden 640, 148,140,154 parameters), fp32 on the T4. Logits are restricted to the 20 canonical letters and renormalised, since the genomes contain nothing else.

Every score is a **masked** likelihood: to score position *i* of a sequence, position *i* is replaced by `<mask>` and the model predicts it from the rest. One sequence of length *L* becomes a batch of *L* rows. This matters — an unmasked pass lets the model see the residue it is being asked to predict.

Two quantities per move:

| | definition | defined for |
|---|---|---|
| **Δlogit** | sum over the positions the move **changed** of `logit(new) − logit(old)`, read from the **base** genome's masked context | the substitution arms only: S1, S2, S3, B (3,192 moves) |
| **ΔPLL** | `PLL(child) − PLL(base)`, where `PLL(s) = Σᵢ log p(s[i] | s masked at i)` | **every** arm, including S4 (all 3,990 moves) |

Δlogit is the decision-relevant one: it is what an operator would consult when deciding whether to propose a substitution, with the context held fixed at the genome it is mutating. The softmax normaliser cancels in the difference, so the canonical restriction does not affect it.

**The indel decision: S4 was not excluded.** All 798 S4 moves change length, so base and child have no position-for-position correspondence and Δlogit is undefined for them by construction (it is stored as `null`). Rather than drop the arm, ΔPLL was added as a second metric: it compares whole-sequence likelihoods and needs no positional alignment, so it covers the indels on the same footing as the substitutions. ΔPLL is also the standard ESM variant-effect score. This is why 6,446 unique genomes had to be scored (3,144 bases plus 3,978 children, deduplicated) rather than only the bases.

Cost: 6,086 s (101 minutes) on an idle T4. The GPU was confirmed empty before the run and released after.

## 1. Δlogit vs realised fitness gain

| subset | n | Pearson r | p~ | Spearman ρ | p~ |
|---|---|---|---|---|---|
| **pooled** | 3192 | +0.025 | 0.16 | +0.013 | 0.45 |
| S1 | 798 | −0.044 | 0.22 | −0.029 | 0.42 |
| S2 | 798 | +0.044 | 0.21 | +0.036 | 0.31 |
| S3 | 798 | −0.012 | 0.73 | −0.015 | 0.66 |
| B | 798 | +0.089 | 0.012 | +0.063 | 0.073 |
| pooled, improved only | 994 | +0.027 | 0.39 | +0.017 | 0.60 |
| pooled, not improved | 2198 | +0.036 | 0.088 | +0.011 | 0.61 |

## 2. ΔPLL vs realised fitness gain

| subset | n | Pearson r | p~ | Spearman ρ | p~ |
|---|---|---|---|---|---|
| **pooled** | 3990 | −0.020 | 0.20 | −0.038 | 0.018 |
| S1 | 798 | −0.082 | 0.020 | −0.074 | 0.037 |
| S2 | 798 | +0.101 | 0.0042 | +0.064 | 0.069 |
| S3 | 798 | −0.049 | 0.16 | −0.069 | 0.050 |
| S4 | 798 | −0.087 | 0.014 | −0.091 | 0.0097 |
| B | 798 | +0.098 | 0.0057 | +0.022 | 0.53 |
| pooled, improved only | 1298 | +0.032 | 0.25 | +0.027 | 0.32 |
| pooled, not improved | 2692 | −0.003 | 0.87 | −0.032 | 0.095 |

## 3. The decision-relevant framing

A correlation coefficient is not how an operator would use the model. The operator asks: *if I propose the moves this model prefers, do more of them work?*

| | Δlogit | ΔPLL |
|---|---|---|
| moves the model **prefers** (Δ > 0) | 1,626, of which 514 improved (**31.6%**) | 1,981, of which 630 improved (**31.8%**) |
| moves the model **dislikes** (Δ < 0) | 1,533, of which 480 improved (**31.3%**) | 1,976, of which 668 improved (**33.8%**) |
| sign agreement (model and fitness move the same way) | 1,567/3,159 = **49.6%** | 1,938/3,957 = **49.0%** |

On Δlogit the gap is 0.3 percentage points. On ΔPLL it runs **the wrong way**: moves the model dislikes improved *more* often than moves it prefers. Sign agreement is a coin flip on both.

For scale, `MOVE_CLASS.md` §3 puts the pooled hit rate at 30.7% (S1) to 38.1% (S4). Sorting moves by ESM-2's preference separates them less than the choice of move class does.

## 4. Stated plainly

**The model's preference does not predict fitness here.** Three readings point the same way:

- The pooled correlations are +0.025 and −0.020 at n ≈ 3,200–4,000 — about 0.05% of variance.
- **The per-arm signs disagree.** On ΔPLL, S1, S3 and S4 are negative while S2 and B are positive. Four of those clear a nominal p < 0.05, but a genuine relationship would not flip sign between arms measured on the same landscape with the same model. Against 20 uncorrected tests, this is what noise looks like.
- Splitting by whether the move improved changes nothing: both subsets are flat.

## 5. Base genome likelihood vs its own fitness

A weaker question, and the only place anything non-zero appears: does a genome ESM-2 finds more likely fold closer to the target? What follows is a pooled association that does not hold in every seed, not a demonstrated effect.

| subset | n | Pearson r | p~ | Spearman ρ | p~ |
|---|---|---|---|---|---|
| **pooled unique (arm,seed,base)** | 3197 | +0.146 | 9.9e-17 | +0.149 | 2.2e-17 |
| S1 | 562 | +0.218 | 1.6e-07 | +0.146 | 0.00051 |
| S2 | 680 | +0.129 | 0.00073 | +0.050 | 0.19 |
| S3 | 617 | −0.101 | 0.012 | −0.078 | 0.053 |
| S4 | 697 | +0.146 | 0.00011 | +0.236 | 2.2e-10 |
| B | 641 | +0.194 | 7e-07 | +0.151 | 0.00012 |
| seed 0 (all arms) | 1063 | +0.058 | 0.061 | +0.036 | 0.24 |
| seed 1 (all arms) | 1081 | +0.160 | 1.2e-07 | +0.108 | 0.00038 |
| seed 2 (all arms) | 1053 | +0.087 | 0.0049 | +0.164 | 8.3e-08 |

Positive in four arms of five (S3 is the exception) and highly significant pooled — but +0.146 is 2% of variance, and **the effect is not present in every seed**. Seeds 1 (+0.160) and 2 (+0.087) carry it; **seed 0 does not** (+0.058, p~0.061 on Pearson and +0.036, p~0.24 on Spearman — not distinguishable from zero). Under the rule used throughout this project, that alone disqualifies it as a demonstrated effect. The pooled significance comes from n ≈ 3,200, not from a result that holds wherever it is checked.

### The breeding-step confound

Bases are drawn from an evolving population, so both quantities drift as the GA advances:

| quantity vs breeding step | n | r | p~ |
|---|---|---|---|
| mean per-position LL | 3197 | +0.118 | 1.8e-11 |
| base TM fitness | 3197 | +0.460 | 6e-174 |

Fitness rises steeply with step, likelihood mildly, so part of the pooled correlation is the GA's own trajectory rather than a likelihood-to-fitness link. Conditioning on step:

| LL vs fitness, within band | n | r | p~ |
|---|---|---|---|
| early 0–5 | 1081 | +0.177 | 3.9e-09 |
| middle 6–11 | 989 | +0.004 | 0.89 |
| late 12–18 | 1127 | +0.173 | 4.8e-09 |
| single step 0 | 203 | +0.193 | 0.0057 |
| single step 9 | 163 | +0.112 | 0.15 |
| single step 18 | 176 | +0.220 | 0.0032 |

Conditioning does not remove it — about +0.17 to +0.22 early and late — so it is not purely a trajectory artifact. But it vanishes in the middle third, +0.2 is still 4% of variance, and it is absent in seed 0 (above), so the residue is small, inconsistent and not established by this project's own rule. Note also that this is a statement about *genomes*, not about *moves*: sections 1–4 show it does not transfer to ranking edits.

### Calibration against the native sequence

- Native 7UR7 chain A, the fitness reference: mean per-position LL **−2.253**.
- The 3,197 unique GA bases: mean **−3.111**, range −3.320 to −2.285.
- **0 of 3,197 bases score above the native sequence.**

Every evolved genome is less likely than the native one under ESM-2, and the best barely reaches it. The whole measurement sits in a narrow band the model considers uniformly unlikely — exactly where its likelihoods are least able to resolve fine differences.

## Scope and limits

- **This cannot show that an ESM-2-*proposed* move would fail.** The distributions differ in kind. An argmax operator proposes the single most-preferred residue at a position — the extreme selected tail of the distribution measured here — and a flat correlation across the bulk is formally compatible with that tail behaving differently. The honest statement is that the screen found no signal where a signal would have been encouraging, not that it proved the arm would fail.
- **One model size.** 150M only. 650M (~2.6 GB fp32) does not fit alongside ESMFold's ~13.8 GB on this card, and 3B less so. Nothing here suggests the flat result is a capacity limit, but nothing rules it out either.
- **A likelihood band below the native sequence.** See the calibration above: the model may simply be poorly calibrated on sequences this far from anything protein-like.
- **One target, one fitness function.** 7UR7 chain A, 63 residues, TM-score — the same scope as every other result in this project.
- **Masked, per-position likelihood only.** No structure-aware or fitness-aware scoring was tried; ESMFold's own confidence (pLDDT, pTM) was not used as a proposal signal.
- **p-values are approximate and uncorrected**, as stated at the top. 20 tests in sections 1–2.
- **Per-position matrices were not saved.** The raw files carry the derived per-move quantities, not the (L, 20) log-probability matrices, which would be ~7.5M floats. A new per-position analysis needs a re-run.

## Files

- `experiments/score_esm2_likelihood.py`: scores every move; writes `results/raw/esm2_likelihood_*.json`. `--reuse` rebuilds the outputs from a previous run's flat dump, skipping the 101 GPU-minutes.
- `experiments/analyse_esm2_likelihood.py`: every table above, into `results/ESM2_LIKELIHOOD_TABLES.md`.
- `results/raw/esm2_likelihood_{S1,S2,S3,S4,B}_seed{0,1,2}.json`: 15 files, 266 moves each. Per move: step, both fitnesses, gain, improved, both lengths, `pll_base`, `pll_child`, `dPLL`, `mean_ll_base`, `mean_ll_child`, `n_changed`, `dlogit` (null for indels). Each file carries the model id, its source move-class file and the native calibration.
- `results/ESM2_LIKELIHOOD_TABLES.md`: the generated tables.
- Inputs, unmodified: `results/raw/move_class_bases_{arm}_seed{n}.json`.
