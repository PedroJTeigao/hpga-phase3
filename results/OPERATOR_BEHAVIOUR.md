# LLM genetic operators comply with the requested format but do not condition on the genome

*A standalone account of one claim, the evidence for it, and its limits. Written to be read without the rest of the repository; source files are named so each number can be checked. No new runs were made for this document: everything below comes from experiments already committed on this branch or on `main` (the latest is `MODEL_HETEROGENEITY_STEP6.md`).*

## The claim

In this project, a language model is used as the crossover and mutation operator of a genetic algorithm over protein sequences. The claim is that these operators **comply with the output format they are asked for, but the choices they make are not driven by the sequences they are given**. What they choose is set by numbers written in the prompt (the cut number in the crossover prompt, the position labels in the mutation prompt) and by the model's own habits, and not, as far as the tests below can reach, by what is in the sequence.

The most direct evidence is the relabelling test in Evidence 3. With the genome held fixed, changing only the position numbers shown to the model moved its favourite position, and the position that had been the favourite (genome index 10) received no choices at all under one relabelling, from either of the two models tested. The favourite is a property of the number shown, not of a place in the genome.

Read this as "as far as tested". It is supported for the two operator prompts studied, one main model (`gemma4:12b`) plus three others on some tests, one seed and mostly 100 calls per condition; the relabelling test covers two models and two label offsets. The limits section says what was not tested.

## Setting, in brief

- **Genome.** A protein sequence, a string of 20 amino-acid letters, length 30 to 80 (mutate is probed at length 63 unless stated).
- **Mutation operator, format `position`.** The model receives the sequence and is asked to change exactly one position: it must answer with one line, `POSITION: <0-62>, NEW: <letter>`. The prompt contains no worked example; the only position numbers in it are the range bounds 0 and 62 (`hpga/sequence_model.py`, `_mutate_position_prompt`).
- **Crossover operator, format `segment`.** The model receives two parents of different lengths and answers with a partition into segments, each labelled with a source parent, with boundaries as percentages of length (`SEGMENTS: 0-40:1, 40-100:2`). The prompt contains a worked example (`0-40:1, 40-100:2`) and a prose sentence ("a boundary at 40 falls 40% of the way along Parent 1 and 40% of the way along Parent 2"). Any integer cut from 1 to 99 is valid, so there are 99 possible values.
- **Fallback.** If the model's reply is invalid after retries (default 2), the operator falls back to the ordinary deterministic operator. "Fallback rate" below is that fraction.
- **Probe conventions.** Every call uses a fresh uniformly random sequence (mutate) or parent pair (crossover), temperature 0.7, a per-call sampling seed drawn from a seeded random stream (seed 0), `num_ctx` 4096. Models are served by a local Ollama server on one GPU. Models used: `gemma4:12b` (11.9B, the reference), `llama3.2:3b`, `qwen2.5:7b`, `mistral:7b`, all Q4_K_M; `llama3.2:1b` appears once.
- **Sources.** `PHASE2_RESULTS.md` (earlier operator studies, on a 2D/3D lattice genome), `PHASE3_RESULTS.md` §9 (the first 20-letter probes and a segment-prompt ablation on `gemma4:12b`), `SEQUENCE_GA_REPORT.md` (the five-arm search comparison), `MODEL_HETEROGENEITY_STEP1..5.md` (the cross-model probes), and the raw logs and generated tables in `results/raw/`.

## Evidence 1. Format compliance is governed by the output format, not by the wording of the instruction

The models can be made to comply with a format, but what fixes non-compliance is changing what the model is asked to produce, not rewording the request.

| observation | model, genome | n | source |
|---|---|---|---|
| Asked to restate the whole mutated sequence ("full" format), the model returned the input unchanged | `gemma4:12b`, lattice, 18 positions | 150 of 150 calls in one run | PHASE2 §4.1 |
| Asked to restate two child sequences, the model returned both parents unchanged when the parents differed | `gemma4:12b`, lattice | 39 of 42 calls (93%) | PHASE2 §4.1 |
| Keeping the format and adding an unambiguous retry hint ("Exactly 1 of the 18 positions must differ, not approximately 1") still left the operator falling back to the deterministic one | same | mutate 7/10, crossover 4/10; in all 7 mutate fallbacks the model returned the unchanged sequence on all 3 attempts | PHASE2 §4.2 |
| Same "approximately 5%" wording, but asking only for the positions that change (diff style) | same | 0 of 18 no-ops, against 150 of 150 for the full format | PHASE2 §4.3 |
| `position` mutate and `segment` crossover: fallback | `gemma4:12b`, lattice, 3 symbols | 0/50 each | PHASE2 §4.3 |
| same | lattice, 5 symbols | 0/20 each | PHASE2 §7.6.1 |
| same | 20 letters, length 63 | `position` 0/400, `segment` 0/100 | PHASE3 §9.1 |
| "full" formats at 20 letters | same | mutate 24/30 (80%), crossover 26/30 (87%) fell back | PHASE3 §9.1 |

Two qualifications from the cross-model probes (`MODEL_HETEROGENEITY_STEP1.md`, `STEP2`, 100 calls per operator per model, 20 letters, length 63):

| model | `mutate/position` fallback | `crossover/segment` fallback |
|---|---|---|
| `gemma4:12b` | 0/100 | 0/100 |
| `qwen2.5:7b` | 0/100 | 0/100 |
| `mistral:7b` | 4/100 | 13/100 |
| `llama3.2:3b` | 41/100 | 2/100 |

- Compliance is also a property of the model: the format does not rescue a model that cannot follow it (`llama3.2:1b` fell back on 20/20 `mutate/position` calls, PHASE2 §7.1) and `llama3.2:3b` fails the mutate format 41% of the time while following the crossover format.
- Compliance is sensitive to prompt content for some models. Removing the worked example and changing the prose number (setting `absent_prose25` below) raised `mistral:7b`'s crossover fallback from 4/100 to 47/100 (`STEP2`); `qwen2.5:7b` stayed at 0/100.

Conclusion for this claim: the output format decides whether the operator complies, provided the model is capable enough; rewording the instruction did not change the outcome where it was tried. This is the "complies" half of the claim.

## Evidence 2. The crossover cut is set by the number in the prompt's prose

Every model's most-used cut is 40, which is the number in the prompt's worked example and prose sentence (`MODEL_HETEROGENEITY_STEP1.md`, 100 calls per model): `gemma4:12b` 40 in 82% of cuts (the rest at 50), `qwen2.5:7b` 100%, `llama3.2:3b` 47% (cuts also at 80, 60), `mistral:7b` 49% (cuts also at 80). The question is whether that number comes from the prompt or from the model.

**Removing every 40 from the prompt.** Setting `absent_prose25` removes the worked example (in the prompt and in the retry hint) and rewrites the prose sentence with 25 in place of 40, so no 40 remains in the template. 100 calls per model and setting, run separately per model:

| model | cut values used, prompt as shipped | cut values used, no 40 in the prompt | cuts equal to 40 in the second setting |
|---|---|---|---|
| `qwen2.5:7b` | 40×100 | 25×100, 75×11, 50×1 | 0 of 112 |
| `llama3.2:3b` | 40×93, 80×82, 60×19, 20×9, 50×3, 70×3, 75×1 | 25×96, 50×96, 75×94 | 0 of 286 |
| `mistral:7b` | 40×95, 80×88, plus 9 others at 1–2 each | 25×53, 50×40, 75×28, 80×22, 45×4, 55×3, 49×2, 70×2, plus 5 at 1 each | 0 of 159 |
| `gemma4:12b` | 40×79, 50×21 | 50×100 (25 never chosen) | 0 of 100 |

(`gemma4:12b` row from `PHASE3_RESULTS.md` §9.3; other rows from `MODEL_HETEROGENEITY_STEP2_SEGEX.md`. `mistral:7b`'s second setting rests on 53 valid declarations because 47 of 100 calls fell back.)

Three things follow. No model keeps 40 when the prose says 25 and no 40 is in the prompt; that includes the calls in which a parent of length 40 is printed as data, none of which produced a 40 cut. (A small residual of 40 does return in the step-6 sweep below, in cells where the prose number is not followed.) In the three models other than `gemma4:12b`, 25, the number now left in the prose, appears in every valid declaration (96/96, 100/100, 53/53), exactly as 40 did before. And `gemma4:12b` behaves differently: it ignores the illustrative number and puts the cut at the midpoint, 50, in 100 of 100 calls, never 25. Its midpoint habit is not caused by a number: the earlier lattice segment prompt contained no illustrative cut value either, and there `gemma4:12b` cut at or one position from the exact midpoint in 40 of 40 calls (`PHASE2_RESULTS.md` §4.4, `PHASE3_RESULTS.md` §9.4).

**Which piece of the prompt carries the number (`qwen2.5:7b` only).** The setting above changes two things at once. With `qwen2.5:7b`, which has no fallbacks, a 2×2 separates them (`MODEL_HETEROGENEITY_STEP3.md`, 100 calls per cell, all cells 0/100 fallback):

| prose sentence says | worked example | cuts |
|---|---|---|
| 40 | present (as shipped) | 40×100 |
| 40 | removed | 40×100 |
| 25 | present (`0-40:1, 40-100:2`) | 25×97, 40×3 |
| 25 | removed | 25×100, 75×11, 50×1 |

The prose number carries the effect. Removing the example changes nothing; changing only the prose number to 25 moves the cut to 25 in 97 of 100 calls. The example keeps a small residual pull (3 of 100 calls stay at 40 when it contradicts the prose). For `gemma4:12b` the same conclusion was reached by a different route: moving the example to 70 left the mode at 40 (74%) and removing it left 40 (77%), while changing the prose number moved it (`PHASE3_RESULTS.md` §9.3, 100 calls per setting). The prose-versus-example split was not separated for `llama3.2:3b` or `mistral:7b`.

**Varying the prose number (step 6, `MODEL_HETEROGENEITY_STEP6.md`).** The 2×2 used only 40 and 25. `qwen2.5:7b` and `gemma4:12b`, 100 calls per model per value, example removed in every setting so only the prose number varies, same length, bounds, temperature and seed. Share of cuts equal to the prose number N (fallback 0/100 in all ten cells):

| prose N | `qwen2.5:7b`: cut = N | modal cut | `gemma4:12b`: cut = N | modal cut |
|---|---|---|---|---|
| 10 | 1 of 107 | 30 (67%) | 0 of 100 | 50 (96%) |
| 25 | 100 of 112 (all 100 calls) | 25 (89%) | 0 of 100 | 50 (100%) |
| 37 | 100 of 100 | 37 (100%) | 27 of 100 | 50 (73%) |
| 60 | 97 of 100 | 60 (97%) | 90 of 100 | 60 (90%) |
| 90 | 31 of 120 | 30 (59%) | 0 of 100 | 50 (100%) |

The prose number carries the cut over part of the range, not all of it. `qwen2.5:7b` follows it at 25, 37 and 60 (the non-round 37 in 100 of 100 calls, so it is not specific to round numbers) and breaks at both extremes: at 10 it names 10 in one call and 30 in 72, and at 90 it names 90 in 31 calls and 30 in 71. The 30 is the lower length bound printed elsewhere in the prompt ("between 30 and 80 letters long"); that as the reason was not tested. `gemma4:12b` does sometimes follow the number, at 60 in 90 of 100 calls and at 37 in 27 of 100, but never at 10, 25 or 90, where it goes to the midpoint (50) in 96 to 100 of 100 calls. The values it follows are within 13 of the midpoint and the ones it ignores are 25 or more away, which fits a window around 50 whose edge these five values do not locate. Where the prose number is ignored, a cut of 40 returns in 3% to 8% of cuts (20 calls in all, only one with a parent of length 40) although no 40 is in the prompt.

For "does not condition on the genome": with the prompt as shipped, `qwen2.5:7b` cut at exactly 40 across 100 different parent pairs, and with the prose number changed it cut at 25 across the same pairs. The cut followed the number written in the prompt, not the parents.

## Evidence 3. The mutation position follows the labels shown to the model, not a position in the genome

**A strongly non-uniform choice.** For `mutate/position` at length 63, `gemma4:12b` chose only 20 of 63 positions in 400 calls, one of them (position 10) in 149 of 400 (37.3%), chi-square 4324 on 62 degrees of freedom, p < 5e-5 (`PHASE3_RESULTS.md` §9.2). The same kind of concentration appeared on the lattice genome: 80% of mutations on positions 0 and 1 of 18 and only 7 positions touched, in 40 calls (`PHASE2_RESULTS.md` §4.4). At 100 calls and length 63 the four models differ (`MODEL_HETEROGENEITY_STEP1.md`): `gemma4:12b` 17 distinct positions, mode 10 (38%); `llama3.2:3b` 21, mode 32 (23.7%, 59 successful calls); `qwen2.5:7b` 6, mode 10 (55%); `mistral:7b` 40, mode 48 (6.2%), the closest to uniform though still non-uniform (chi-square 101, p = 0.0016).

**No worked example to copy, but the numbers shown matter.** Unlike the crossover prompt, the mutate prompt has no worked example or illustrative index; its only position numbers are the range bounds 0 and 62 and the labels the model answers with. The modal positions (10, 32, 10, 48) are none of the bounds. The relabelling test below shows that the numbers the prompt presents nevertheless decide the favourite.

**The favourite does not scale with length.** `gemma4:12b` and `qwen2.5:7b`, 100 calls at each of three genome lengths (`MODEL_HETEROGENEITY_STEP4.md`):

| model | length | modal position (share) | choices on a multiple of 10 (uniform would give) |
|---|---|---|---|
| `gemma4:12b` | 40 | 10 (38%) | 39% (10%) |
| `gemma4:12b` | 63 | 10 (38%) | 41% (11%) |
| `gemma4:12b` | 80 | 10 (42%) | 44% (10%) |
| `qwen2.5:7b` | 40 | 0 (54%); position 10 second (29%) | 83% (10%) |
| `qwen2.5:7b` | 63 | 10 (49%); position 0 second (25%) | 74% (11%) |
| `qwen2.5:7b` | 80 | 10 (67%); position 0 second (28%) | 97% (10%) |

A relative-position effect would have moved `gemma4:12b`'s mode to about 6.3 at length 40 and 12.7 at length 80; it stayed at 10. `qwen2.5:7b`'s favourites are positions 0 and 10 at every length; which of the two leads changes (0 at length 40, 10 at 63 and 80), but neither scales with length. Multiples of 10 take 39% to 97% of choices across the six cells, against 10% to 11% under uniform choice. Low indices dominate: at length 80 `gemma4:12b` never chose a position above 34 and `qwen2.5:7b` never above 20. In these runs the favourite is a fixed label; the relabelling test below shows it is the label, not a place in the genome.

**Relabelling the positions with the genome unchanged (the most direct test).** `gemma4:12b` and `qwen2.5:7b`, 100 `mutate/position` calls each at length 63, temperature 0.7, seed 0, in three labellings of the same sequences (`MODEL_HETEROGENEITY_STEP5.md`): the shipped 0-62; 100-162; and 107-169, added so that the round label 110 (genome index 10 under the 100 offset) and the old genome index 10 (label 117 under the 107 offset) could be told apart. The prompt states the range, the format line and the bounds consistently ("positions labelled 100-162, the first letter is position 100", `POSITION: <100-162>`), and the reply is mapped back before it is applied; a reply outside the stated range is rejected and logged.

| model | labels shown | modal label (share) | choices at the label of the old genome index 10 | fallback |
|---|---|---|---|---|
| `gemma4:12b` | 0-62 (shipped) | 10 (38%) | 38 of 100 (label 10) | 0/100 |
| `gemma4:12b` | 100-162 | 101 (25%); 104 18%, 100 16%, 103 16% | 8 of 100 (label 110) | 0/100 |
| `gemma4:12b` | 107-169 | 107 (20%) tied with 110 (20%) | **0 of 100 (label 117)** | 0/100 |
| `qwen2.5:7b` | 0-62 (shipped) | 10 (55%) | 55 of 100 (label 10) | 0/100 |
| `qwen2.5:7b` | 100-162 | 100 (96%); 101 4% | 0 of 100 (label 110) | 0/100 |
| `qwen2.5:7b` | 107-169 | 108 (99%); 110 1% | **0 of 100 (label 117)** | 0/100 |

**The key fact.** Under the 107-169 labelling the old genome index 10 (label 117) received 0 choices from both models, having taken 38% (`gemma4:12b`) and 55% (`qwen2.5:7b`) under the shipped labelling. The genome and the index were unchanged; only the number shown differed. The favourite is a property of the number shown to the model, not of the position in the genome. Both models also followed the stated labels: 0 of 100 first replies fell outside the stated range in every cell, so there were no unshifted answers such as a bare 10.

**No simple rule covers both labellings.** A preference for the start of the range predicts position 0 under the shipped labelling, but 10 beats 0 there (`gemma4:12b` 38% against 1%, `qwen2.5:7b` 55% against 11%), so it fails under 0-62. A preference for round numbers predicts 110 or 120 under 107-169, but `qwen2.5:7b` goes to 108 (99%, with 110 at 1%), and under 100-162 the round label 110 gets 8% (`gemma4:12b`) and 0% (`qwen2.5:7b`), so it fails under the offsets. A fixed genome index fails because index 10 gets nothing under 107-169. What is observed is that under each offset labelling both models concentrate on a few labels near the start of the stated range (`qwen2.5:7b` puts 96% on 100 under 100-162 and 99% on 108 under 107-169; `gemma4:12b` puts 90% on labels 100-105 under 100-162 and 92% on labels 107-115 under 107-169), but the exact label is not given by a single rule, and the shipped labelling shows 10 rather than 0. One candidate, untested, is that 10 is a habit attached to a 0-based range and low labels play the analogous role in a 100-based range. For `gemma4:12b` under 107-169, label 110 ties for the mode; it is also genome index 3, a favourite under the shipped labelling (12%), so the result does not show a round-number effect, but a round-label contribution for that model cannot be excluded.

What this adds to the claim: relabelling changes nothing about the sequence, yet it changes which position is chosen, and a position that was chosen 38% to 55% of the time under one labelling is chosen never under another. The choice is therefore not driven by the genome, at least for these two models and these two offsets. Together with Evidence 4 (the letter at the chosen position does not predict the choice) this is the mutation-side evidence for the claim.

**Inside the search.** In the five-arm comparison's LLM arm, over 5 seeds, the most-used mutation index was 10 in three seeds (12%, 12%, 13% of changed positions), 3 in one (17%) and 40 in one (12%), on genomes of varying length between 30 and 80, with 26 to 41 distinct positions used per run (`SEQUENCE_GA_REPORT.md` §3 table 9, §4).

## Evidence 4. The letter at the chosen position does not predict the choice

An offline test on call logs that already existed, with no new calls (`MODEL_HETEROGENEITY_STEP4.md`). For each `mutate/position` call I read the genome from the prompt, the position from the first valid reply, and the letter sitting there, and compared the letters at chosen positions with the letters available in the same genomes by Monte Carlo chi-square (20,000 draws) under two nulls: **A**, position uniform over the genome; **B**, the model's own chosen positions kept but paired with genomes from other calls, which preserves the positional bias and breaks any link to the content. Rejecting B would mean the choice depends on the letter. The same test was run on four residue classes (hydrophobic AVILMFW, charged DEKRH, polar STNQCY, special GP).

| source | calls used | letters vs A, p | letters vs B, p | classes vs A, p | classes vs B, p |
|---|---|---|---|---|---|
| `gemma4:12b` | 100 | 0.92 | 0.74 | 0.79 | 0.92 |
| `gemma4:12b` (earlier run) | 400 | 0.058 | 0.035 | 0.52 | 0.60 |
| `llama3.2:3b` | 59 | 0.23 | 0.24 | 0.77 | 0.59 |
| `qwen2.5:7b` | 100 | 0.43 | 0.72 | 0.24 | 0.34 |
| `mistral:7b` | 96 | 0.92 | 0.92 | 0.66 | 0.80 |

No detectable dependence on the letter in any model, including at n = 400. The one nominal p below 0.05 (n = 400, letters, null B, p = 0.035) is one of ten letter-level tests, uncorrected, with an incoherent pattern (C and T chosen 10 times each against about 18 expected, N, V and W chosen 26 to 28 times against 17 to 20), and the class-level test on the same data is null (p = 0.60). For that run hydrophobic residues are 35.2% of the available letters and 36.8% of the letters at chosen positions. The sequences in this test are uniformly random, so there is no designed structure for an operator to exploit; what the test shows is only that the choice does not track letter identity.

## What this implies for the five-arm comparison

The five-arm search comparison (`SEQUENCE_GA_REPORT.md`) compared random search, GA with deterministic operators, GA with these LLM operators, and two variants with injected genomes, over 5 seeds at equal numbers of distinct folds. Its result was: the LLM-operator GA beat random search in 5 of 5 seeds (mean difference +0.0862 in TM-score) but was not distinguishable from the GA with deterministic operators (higher in 3 of 5 seeds, mean +0.0234), and with 5 seeds nothing can reach p < 0.05.

What the operator findings add, stated with care:

- The in-loop operator behaviour matches the isolated probes: every accepted crossover in every seed used one of two cut values, 40 or 50 (40 in 85% to 95% of them), out of 99 possible, and mutation concentrated on a few favourite indices (above). In that arm the LLM is best described as supplying a narrow, content-blind pattern of perturbations, not as reading the sequences.
- That is consistent with the absence of a demonstrable difference between the LLM-operator arm and the deterministic-operator arm. It does not explain it: no experiment here compared the two arms' operators against each other, so the consistency is a reading, not a finding.
- It does not explain why the LLM-operator arm beat random search either. The five-arm report notes that its improvement over random search cannot be attributed to any operator (no ablation of selection, crossover or mutation), and that remains true.
- It does not show that the LLM operators are worse than deterministic ones, and it does not show that no model or prompt could condition on the genome. Everything was measured on one prompt family.
- The circle-agent proposals in the variant with circles use the same position-edit format but were not tested here for dependence on the genome. Their positions were more spread (40 to 51 distinct indices per run, most-used index 6% to 9%) and a proposal beat its own base genome in 43% to 53% of slots, which the report itself reads as a coin flip (`SEQUENCE_GA_REPORT.md` §9.10, §9.11). This document's claim is about `mutate/position` and `crossover/segment`.
- The cut result had a specific consequence, tested in step 6: varying the prose number varies the cut over part of the range (`qwen2.5:7b` 25 to 60, `gemma4:12b` 37 and 60), not over all of it. Outside that range the cut goes to 30 (`qwen2.5:7b`) or to the midpoint (`gemma4:12b`), still without reading the parents. A search that used a different prose number would therefore steer the cut only inside that range, per model.

## Limits, stated plainly

- **Single seed, mostly 100 calls per cell.** Except for the earlier `gemma4:12b` runs (400 `mutate/position` calls in `PHASE3_RESULTS.md` §9.2, and 40 to 50 calls per condition in the lattice studies of `PHASE2_RESULTS.md` §4), each condition is one run of about 100 calls at seed 0, temperature 0.7. A rerun of `qwen2.5:7b` at length 63 with the same seed and prompt gave a different position count for position 0 (11 vs 25 of 100) because a single retry shifts the shared random stream; the mode and the set of positions were the same. Differences of that size between runs should not be read as effects.
- **Round-number and low-index explanations are now partly resolved, not fully.** The relabelling shows that neither a fixed genome index nor round numbers alone explain the favourite, and that no simple rule covers both the shipped and the offset labellings ("start of the range" fails under 0-62, where 10 beats 0; "round numbers" fails under 107-169, where `qwen2.5:7b` goes to 108). It does not say what does hold in general. For `gemma4:12b` the tie of label 110 with the start label under 107-169 leaves a round-label contribution open.
- **The low-index picture is for two models.** The length sweep and the relabelling covered only `gemma4:12b` and `qwen2.5:7b`. At length 63, `llama3.2:3b`'s mode is position 32 and `mistral:7b`'s is 48 with a nearly flat distribution, so the "low indices dominate" description does not apply to them, and their behaviour across lengths and under relabelling was not measured.
- **Relabelling: two models, two offsets, one seed.** Only offsets 100 and 107 with contiguous labels were tried; no other offset, no non-contiguous or shuffled labelling. Three-digit labels also change the numerals in the answer, so a preference for the start of the range cannot be separated from a preference for shorter or lower-valued numerals. 100 calls per cell.
- **Prose versus example was separated for `qwen2.5:7b` only** (and, by a different route, for `gemma4:12b`). For `llama3.2:3b` and `mistral:7b` the only ablation removed both at once.
- **The prose number was tried at 40, 25, and (step 6) 10, 37, 60, 90, on two models; the example at 40 and 70 only.** The result is a range, not a rule: `qwen2.5:7b` follows the number at 25, 37 and 60 (89% to 100% of cuts) and not at 10 (1%) or 90 (26%); `gemma4:12b` follows 60 (90%) and partly 37 (27%) and not 10, 25 or 90 (0%). Not located: where between 60 and 90 `qwen2.5:7b` starts to fail, where between 25 and 37 and between 60 and 90 `gemma4:12b`'s window ends, and whether extremes fail because they are extreme or because they are far from 50 (10 and 90 confound the two). Why `qwen2.5:7b` goes to 30 at the extremes (a candidate: 30 is the length bound in the prompt) and where `gemma4:12b`'s residual 40 comes from were not tested. `llama3.2:3b` and `mistral:7b` were not swept. Single seed, 100 calls per cell, same 100 parent pairs in every cell of a model.
- **The letter test covers one thing.** It tests the letter at the chosen position only. It does not test neighbouring letters, local motifs, or the replacement letter the model chooses, and it uses uniformly random sequences. Statistical power is modest at about 5 expected calls per letter (n = 100); the class-level tests and the n = 400 run are the most sensitive and show nothing.
- **Crossover conditioning on the parents was not tested directly.** The evidence is that the cut follows a number in the prompt. Whether, for example, `gemma4:12b`'s choice between 40 and 50 depends on the parents' lengths was not analysed.
- **One prompt family, one temperature, one task.** All prompts are the project's own (`position`, `segment`), at temperature 0.7, on random uniform sequences; no other prompt design, temperature or sequence distribution was tried. Models are small to mid-size quantised open models (3B to 12B).
- **Different genome types across sources.** Evidence 1 partly rests on the earlier lattice studies (3 and 5 symbols); the 20-letter results repeat the pattern but are separate measurements.
