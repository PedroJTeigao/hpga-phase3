# Phase 3 Results — Circles, the Blackboard, and the Diversity Gap

*Standalone results summary, same structure as `Phase 2/results/PHASE2_RESULTS.md`
(itself modeled on `Phase 1/results/PHASE1_RESULTS.md`). Phase 3 has no
separate code folder — `hpga/circles.py` and `hpga/blackboard.py` live
inside this project's `hpga/` alongside the Phase 2 modules. See
`hpga/circles.py`'s module docstring for why this is a new architecture,
not an extension of Phase 2's `hpga/agents.py` (§8 of `PHASE2_RESULTS.md`):
that document's three §8 results — communication didn't pay for itself,
the diversity effect came from the roles rather than the channel, and
passed folds carried no reasoning the receiver couldn't derive itself —
constrain this design rather than being inherited machinery from it. §7-8
of `PHASE2_RESULTS.md` predate this document and are referenced below, not
superseded.*

## 1. What was built and measured

`hpga/circles.py` groups `HPGA_AGENTS_PER_CIRCLE` agents into
`HPGA_N_CIRCLES` circles. Agents within a circle may consult each other
(`_consult`, one call per circle) before proposing a genome; agents in
different circles cannot — `consult()` never touches the blackboard, which
makes that isolation true by construction rather than convention.
`hpga/blackboard.py` is the only cross-circle channel, and by design it
carries *observations* (free-text reasoning about a fold), never
*candidates* (genomes) — the one thing explicitly not repeated from §8's
finding that passed folds carried no reasoning the receiver couldn't
derive by evaluating the fold itself. Same additive-population contract as
`agents.py`: `run_circles()` always returns exactly
`n_circles * agents_per_circle` genomes appended after
`next_generation()`'s ordinary fill-to-`pop_size` loop, never carved out of
it, so a circles-on run never issues fewer top-level LLM calls than the
baseline — which is what would confound a fitness comparison by call
budget alone. The central circle's directive is broadcast identically to
every circle (global, not per-circle targeting).

Four things were measured, all from the anchor run below unless noted:
population diversity per generation (§2), the identical-parent rate
underlying a candidate explanation for part of that gap (§3), a
crossover-correctness redesign that came out of fixing the operators used
in this run (§4), and fitness (§5).

**Anchor run**: `experiments/run_circles_smoke.py`, revision 4 — the
fourth iteration of this experiment, each revision a direct response to
what the previous one's numbers showed (see the script's docstring for the
full chain; §3 below recaps the part of it that matters for the
identical-parent result). `results/raw/circles_smoke_summary_1789603059.json`.
Config: `PROMPT_STYLE="best"` (§4.3 of `PHASE2_RESULTS.md`'s `position`
mutate + `segment` crossover simultaneously — both operators at their most
reliable, near-0% fallback), `genome_length=18`, `pop_size=8`,
`n_generations=10`, `N_CIRCLES=2`, `AGENTS_PER_CIRCLE=2`, seed=0, a single
sequence (`make_timing_sequence(length=20, seed=1)`).

**"Revision 4," precisely — checked against file timestamps, not assumed.**
No revision numbered 5 exists anywhere in this codebase (`grep -rn
"[Rr]evision 5"` over every `.py` file returns nothing) — the numbering
genuinely is 4, confirming the script's own docstring rather than this
document's earlier draft. But two distinct states both answer to that name,
and conflating them is what produced the inconsistency this section
originally had. In order: `run_circles_smoke.py` was last edited
2026-09-16 19:44 — that version's own inline comment on `PROMPT_STYLE="best"`
reads `# ...; CROSSOVER_MIN_DIFF untouched`. `experiments/calibrate_crossover_min_diff.py`
ran at 19:45 (§4's 17.6% false-rejection finding). `hpga/operators.py` was
then independently edited at 19:56 to drop `CROSSOVER_MIN_DIFF` for
`segment`-style crossover entirely (§4's redesign) — *after* the script
above was finalized, *before* the anchor run itself, which starts at
19:59:35. So the run that actually produced this document's numbers ran
under the *post-redesign* `operators.py`, not the pre-redesign state its
own script comment still describes; that comment went stale the moment
`operators.py` changed under it. The redesign is a fix to a shared
dependency the script imports, not a change to `run_circles_smoke.py`'s
own parameters, which is presumably why it was never given its own
revision number — but it is real, it is why this run's crossover fallback
is not fully explained by `PROMPT_STYLE="best"` alone, and it is what
distinguishes "revision 4 as written" from "revision 4 as it actually
ran."

## 2. The diversity gap is the finding

Every diversity-logged run in this project so far — five before this one,
plus this one — shows the same qualitative result: population diversity
(mean pairwise Hamming distance) is higher with the LLM-agent architecture
on than off, by a wide margin, regardless of which operators were broken
at the time:

| run | config | operator health underneath | gens matched | mean off | mean on | **ratio** |
|---|---|---|---|---|---|---|
| `agents_smoke_1788633787` | `hpga/agents.py`, genome_length=8, A=2 (explore+refine) | free-form role prompts, no affordance fix (`PHASE2_RESULTS.md` §8.3: refine is a self-sustaining no-op) | 5/5 | 1.66 | 5.24 | **3.16x** |
| `agents_smoke_1789504595` | same config, repeat | same | 5/5 | 1.98 | 4.47 | **2.26x** |
| `agents3_1788635605` off/nocomm | `agents.py`, genome_length=12, A=2 | same free-form roles; underlying crossover/mutate at 93/100 requests retried, 29 failures (`agents_3arm_summary_1788635605.json`) | 6/6 | 4.72 | 7.77 | **1.65x** |
| `agents3_1788812113` off/nocomm | same config, repeat | same | 4/6 (**on-arm log truncated at gen 3** — flagged, not silently used) | 5.60 | 8.12 | **1.45x** |
| `agents3_1788814924` off/nocomm | same config, repeat | same | 6/6 | 4.72 | 7.72 | **1.64x** |
| **`circles_1789603059` off/on (this run)** | `hpga/circles.py`, genome_length=18, N_CIRCLES=2x2 | `PROMPT_STYLE="best"`: crossover/mutate near-0% fallback | 10/10 | 5.06 | 11.28 | **2.23x** |

(`off` is bit-identical across the three `agents3` repeats — same seed, no
agents, fully deterministic selection — which is why its mean is exactly
4.72 in all three; not a bug, a property of the control arm.)

Ratios range 1.45x-3.16x across six independent runs spanning genome
lengths 8-18, population sizes 5-8, and operator reliability from the
free-form `agents.py` roles (majority-fallback, never given a
position/segment-equivalent affordance — see `circles.py`'s own "known
open risk" note) to this run's fully-fixed `best`-style operators. **The
gap does not depend on the operators working.** That is unusually
well-tested for a single-seed result (§5), and it is the load-bearing
finding of this document — everything below either qualifies it or asks
what it bought.

## 3. What doesn't survive: the identical-parent "feedback loop"

A candidate mechanism for part of the diversity gap: if crossover
disproportionately draws identical parents in the `off` arm (a population
converging faster without circles feeding it fresh material), then
`CROSSOVER_MIN_DIFF`-gated crossover becomes structurally more likely to
fail there too, compounding the gap rather than just reflecting it. This
was tracked directly — `hpga/operators.py`'s `_llm_crossover` logs
`identical_parents` on every attempt (added in circles revision 3
specifically to make this observable per generation per arm).

The hypothesis does not survive being followed to a working-operator
regime:

| revision | crossover / mutate state | mean identical-parent rate, off vs. on | **ratio** |
|---|---|---|---|
| revision 3 (`run_circles_smoke.py` docstring) | crossover fixed (`segment`, still `CROSSOVER_MIN_DIFF`-gated at that point); mutate still broken (`full`, 88-93% fallback at genome_length=18) | 0.54 vs. 0.20 | **2.7x** |
| *(unretained — see below)* | provenance unknown — not attributable to any specific revision | reported around 1.65x at some point | *not verifiable* |
| revision 4 / this run (`circles_smoke_summary_1789603059.json`) | both fixed (`best`: `position` mutate + `segment` crossover, `CROSSOVER_MIN_DIFF` dropped for `segment` by §4's redesign, ~0% fallback) | 0.200 vs. 0.167 (unweighted mean across 10 generations) | **1.2x** |

The two endpoints are independently sourced and reproducible: 2.7x is
quoted directly from `run_circles_smoke.py`'s revision-3 docstring (a
documented result from a real run, not re-derived here); 1.2x is
recomputed directly from this run's `off_identical_parent_rate`/
`on_identical_parent_rate` arrays (`0.2/0.16667 = 1.2` exactly, unweighted
mean of per-generation rates — a calls-weighted version gives 1.51x, still
well down from 2.7x, so the direction doesn't depend on that choice).

**The middle point is a known gap, not papered over — and it's narrower
than this document previously claimed.** A ~1.65x figure was reported at
some point before this write-up, but no run in `results/raw/` retains the
ops log it would have been computed from, and — checked directly against
file timestamps for this revision — there is no time window for a
separate completed circles-smoke run to have produced it either: the
script was finalized at 19:44, the crossover-validation calibration ran at
19:45, `operators.py` was redesigned at 19:56, and the anchor run's own
off-arm log starts at 19:59:35, all in one continuous session with no gap
for an intermediate 10-generation run in between. So the earlier framing
of this figure as coming from "an intermediate configuration during
revision 4" was not supportable and has been removed — it should be read
as **provenance unknown**, not just unretained. The only files with real
(non-fixture) `identical_parents` data at this project's timestamps are
this run's off/on logs. It is reported here as *reported, not verified*,
and excluded from the trend line for that reason. This is the same
failure mode this project's version-control situation created generally
(until the fix described in the note at the end of §5): a number that
exists only in a conversation and not on disk cannot be checked later —
and, as this correction shows, can also drift in *when it's claimed to
have happened*, not just in whether it can be re-verified.

Dropping the middle point costs nothing the argument needed: **2.7x
collapsing to 1.2x once both operators stopped failing is enough on its
own.** A real hypothesis at the broken-operator stage becomes, at best, a
small residual effect once the operators are reliable — not zero (1.2x is
still `off` slightly favoring identical parents more than `on`), but no
longer large enough to carry the explanatory weight it looked like it had
in revision 3. It should be read as **a hypothesis that looked strong and
shrank**, not as a demonstrated mechanism behind §2's gap.

## 4. The crossover-validation redesign

A result in its own right, independent of circles specifically.
`CROSSOVER_MIN_DIFF=2` (reject-and-retry unless each child differs from
*both* parents by ≥2 positions) was the correctness gate for `full`/`diff`
-style crossover since Phase 2 §4.2. `experiments/calibrate_crossover_min_diff.py`
measured it against 5000 genuine deterministic crossovers (random parent
pairs, genome_length=18, 5-symbol alphabet — the best case for how
different two parents can be, no model involved) and found **the threshold
rejects 17.6% of correct, deterministic output outright**
(`results/raw/crossover_min_diff_calibration.json`). Raising the threshold
doesn't fix it: a legitimate edge-adjacent cut and a near-echo dodge
produce the *same* diff count (both can differ from one parent by as
little as 1 position), so no single value of this metric separates a bad
answer from a good one — the check was measuring the wrong thing, not
miscalibrated on the right thing.

`segment`-style crossover (§4.3 of `PHASE2_RESULTS.md`) sidesteps the
problem structurally rather than retuning it. The model declares its own
cut (`SEGMENTS: 0-8:1, 9-17:2`) and the child is *built from that
declaration in code*, not produced independently and then compared
against it — matching the declaration is true by construction. What
actually needs validating is the declaration's structural validity (exact
coverage, ≥2 segments, both parents used), which `_llm_crossover` now
checks instead of the post-hoc diff-magnitude threshold. No new threshold,
no calibration free parameter — it uses information already present in
the model's own response and falls back to the deterministic operator on
a malformed declaration, exactly as before. `full`/`diff`-style crossover
still has nothing to validate against but the output itself, so
`CROSSOVER_MIN_DIFF` remains their fallback gate, uncalibrated, exactly as
it was (`hpga/operators.py` lines 94-122).

**This generalizes past this project**: when a correctness check is
validating a free-form output against a threshold on the output itself,
and the generator could in principle be made to declare its own
construction plan, checking the plan's structural validity is a strictly
different (and here, strictly better) kind of check than thresholding the
result — it stops conflating "differs enough to look real" with "is
real," which is what the 17.6% false-rejection rate was actually
measuring.

## 5. Limitations

- **Circles have not won on fitness, in any of four comparisons in this
  run.** Diversity is a means, not the objective, and at this scale the
  architecture has not converted one into the other:

  | comparison | off | on (circles) | circles wins? |
  |---|---|---|---|
  | final best fitness (gen 9) | 3.0 | 3.0 | no — tie |
  | generation the final value is first reached | gen 4 | gen 3 | marginal (1 generation, single seed) |
  | total fitness gain (gen 0 -> gen 9) | +2.0 | +2.0 | no — tie |
  | fitness gain per 1k tokens | 0.0659 | 0.0312 | **no — circles is ~2.1x less token-efficient** |

  This is a real contrast with `PHASE2_RESULTS.md` §8's `agents.py`
  result, not a repeat of it: there, `nocomm`/`comm` reached fitness 2.0
  against `off`'s flat 1.0 — the role architecture *did* move fitness.
  Circles, on the same kind of paired comparison, has not, in this run or
  in the four ways of reading it above. §2's diversity gap is real and
  robust; it has not yet been shown to buy anything on the objective the
  GA actually optimizes.
- **Single seed, ten generations, one problem.** Every number in §2's
  `circles` row and all of §3-4 comes from one run: seed=0, one
  `make_timing_sequence` sequence, `n_generations=10`. The five prior §2
  runs add independent seeds/configs for the *diversity-gap* finding
  specifically, which is why that finding is reported with more
  confidence than §5's fitness comparison or §3's 1.2x point, neither of
  which has been repeated.
- **The raw-data retention gap flagged in §3 was a process problem, not
  just a footnote — and it has since been fixed at the infrastructure
  level, not just noted.** This whole project previously lived
  unversioned on `/scratch`, which is wiped after 28 days with no backup;
  the 1.65x figure is a direct casualty of that (a number that existed
  only in a conversation, never landed in `results/raw/`, and couldn't be
  checked again). As of this write-up the project has `git` history (this
  file's own commit included), a mirror on the 50GB backed-up
  `/afs/ece.cmu.edu/usr/pcanaste` volume, and a private GitHub remote
  (`PedroJTeigao/hpga-phase3`) — three independent copies. That stops this
  *class* of loss; it does not retroactively recover the 1.65x point, and
  it doesn't substitute for treating "save the raw log" as part of
  finishing a run, which remains a discipline this project has to keep
  applying run by run, not something the infrastructure fix does for it.

## 6. What Phase 3 establishes, and what's next

Phase 3 establishes one robust result and one important negative result,
and they don't cancel — they define what's actually open.

The robust result (§2): circles/agent architectures on this substrate
reliably produce more diverse populations than the plain baseline, by a
wide margin (1.45x-3.16x across six independent runs), and — unlike
almost every other number in this project's operator-reliability work —
that result does not depend on the underlying LLM operators being
reliable. It held when mutate and crossover were mostly falling back to
the deterministic operator (the `agents_smoke`/`agents3` runs) and it
holds now that both are fixed (`best`-style, this run). A candidate causal
story for part of that gap — the identical-parent feedback loop — was
tested directly and did not survive: it was strong (2.7x) precisely when
an operator was still broken, and shrank to a small residual (1.2x) once
both were fixed (§3). That is the right order of events for ruling a
mechanism *out*, not in: if identical parents were doing real work, fixing
the operators that fed them shouldn't have made the effect smaller.

The negative result (§5) is what keeps §2 from being read as a win on its
own: circles has not moved fitness relative to its baseline in any of four
comparisons in this run, and on the token-efficiency comparison
specifically it is clearly worse. Contrasted directly with
`PHASE2_RESULTS.md` §8, where the plain role architecture *did* move
fitness (1.0 -> 2.0) — this is not "LLM agents never help here," it's
"this specific architecture, at this scale, hasn't yet." Diversity is
necessary for a GA to keep finding new material, but this document has
not shown it's sufficient here; the open question §2 raises and §5 leaves
unanswered is whether a longer run, a larger population, or a harder
landscape (where a converged, low-diversity `off` population would stall
below a peak that a higher-diversity `on` population could still reach)
would let this gap convert into a fitness advantage, or whether the extra
tokens circles spends are simply not buying anything on this problem at
this scale.

Next:
- **Repeat the circles anchor run at more seeds.** §2's confidence rests on
  five independent prior runs plus this one; §3's 1.2x and all of §5's
  fitness comparisons rest on this run alone. The same multi-run treatment
  that made §2 credible hasn't been applied to the other three findings
  yet.
- **Test whether the diversity gap converts to fitness on a harder
  landscape.** This run's sequence reaches its ceiling by generation 3-4
  in both arms — not enough runway to distinguish "diversity doesn't help
  here" from "neither arm needed it." A longer sequence or more
  generations, chosen the way `run_circles_smoke.py`'s revision 1 fix
  chose `pop_size`/`n_generations` (deterministically confirmed to still
  show movement before spending LLM budget on it), is the direct test.
- **Recalibrate `CROSSOVER_MIN_DIFF` for `full`/`diff`-style crossover, or
  retire those styles.** §4's redesign only fixed `segment`; the old
  17.6%-false-rejection threshold is still the fallback gate for the two
  styles that still restate letters independently.
- **Keep the raw-log discipline the retention-gap fix (§5) doesn't
  automate.** Version control and backups now exist; they only help if
  every run that produces a real number also commits its `results/raw/`
  output before that number gets reported anywhere.
