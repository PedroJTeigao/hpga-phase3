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

## 6. Testing whether the diversity gap converts to fitness, at a configuration with headroom

*Follow-up to the "next steps" that closed this document before this
section existed (now §7). The anchor run (§1-5, genome_length=18) plateaus
by generation 3-4 in both arms — "diversity didn't help" there is
indistinguishable from "neither arm needed it." This section reports the
direct test at a configuration chosen specifically to avoid that.*

### 6.1 Picking a configuration, for free

`experiments/sweep_diversity_config.py` (deterministic,
`HPGA_OPERATOR_MODE=deterministic`, zero LLM calls, 37s total) swept
genome_length in {18, 24, 30, 36, 42} at pop_size=8 against a pop_size=64
reference, 150-generation horizon, 3 seeds. `genome_length=24` was picked:
pop=8 settles ~4.17 fitness points below pop=64's ceiling (real trapping —
its own plateau doesn't arrive until generation ~54 on average) and is
still climbing at every checkpoint from generation 10 through 25 in that
free proxy. `pop_size=8` and the circles architecture (`N_CIRCLES=2`,
`AGENTS_PER_CIRCLE=2`, `PROMPT_STYLE="best"`, `CENTRAL_MODE=llm`) were kept
identical to the anchor run — only the sequence (genome_length 18 -> 24,
`make_timing_sequence(length=26, seed=1)`) and seed count (1 -> 3) changed.
`experiments/run_circles_diversity_fitness.py`,
`results/raw/circles_diversity_fitness_summary_1789650346.json`.

### 6.2 The diversity gap: this is the headline result, not the fitness

| gen | off (mean) | on (mean) |
|---|---|---|
| 0 | 18.92 | 18.92 |
| 1 | 13.64 | 16.76 |
| 2 | 6.18 | 14.75 |
| 3 | 3.33 | 13.35 |
| 4 | 2.08 | 12.89 |
| 5 | 1.93 | 12.34 |
| 6 | 1.99 | 13.21 |
| 7 | 2.25 | 13.63 |
| 8 | 2.44 | 12.69 |
| 9 | 2.18 | 12.97 |

Final-generation ratio: 12.97 / 2.18 = **5.95x** — against the 1.45x-3.16x
range across all six runs reported in §2. The gap didn't just replicate on
a harder landscape, it **widened**, by roughly 2x over the widest prior
run. This is the strongest version of §2's finding produced so far, and it
came from a run designed specifically to give the comparison a fair chance
to fail, not to succeed.

### 6.3 Fitness: circles' first non-loss across five comparisons — not a win

| gen | off (mean, [min-max]) | on (mean, [min-max]) |
|---|---|---|
| 0 | 2.00 [1-3] | 2.00 [1-3] |
| 1 | 2.00 [1-3] | 2.00 [1-3] |
| 2 | 2.00 [1-3] | 2.33 [2-3] |
| 3 | 2.33 [1-3] | 2.33 [2-3] |
| 4 | 2.33 [1-3] | 2.33 [2-3] |
| 5 | 2.33 [1-3] | 2.67 [2-3] |
| 6 | 3.00 [2-4] | 2.67 [2-3] |
| 7 | 3.33 [2-4] | 3.33 [2-5] |
| 8 | 3.33 [2-4] | 3.67 [2-5] |
| 9 | 3.33 [2-4] | 3.67 [2-5] |

Per-seed final fitness: off = [2.0, 4.0, 4.0] (mean 3.33); on = [2.0, 4.0,
**5.0**] (mean 3.67). Counting §5's four comparisons (all ties or losses)
plus this run's aggregate final-fitness comparison as a fifth, **this is
circles' first non-loss on fitness in this project** — mean fitness edges
up (3.33 -> 3.67) and one of three seeds ends strictly ahead. It is
deliberately not written as a win: two of three seeds tie exactly, and the
margin on the one that doesn't is a single fitness point.

The seed that produced the higher final value (seed 2: off=4.0, on=5.0) is
also the seed with the smallest token-efficiency deficit (1.51x, against
~2.5x on the other two — §6.4). That pattern — circles pulled ahead where
it was *cheapest* relative to `off`, not where it visibly did something
different — is a more honest read than "circles found extra fitness here":
it is at least as consistent with ordinary run-to-run variance in how much
`off` itself happened to struggle on that seed as with anything
circles-specific.

**A correction, checked directly against this run's raw data rather than
asserted.** A mid-run progress report (seed 0 only, both arms already
complete at the time) described `on` reaching fitness 2.0 at generation 2
against `off`'s generation 6, read as circles reaching the same ceiling
faster. Re-checked now against the same arrays
(`off`=[1,1,1,1,1,1,2,2,2,2], `on`=[1,1,2,2,2,2,2,2,2,2]): **that specific
claim was correct** — both trajectories were already final when it was
made, nothing changed on a second look. But it should not have been read
as an advantage, and isn't carried into this write-up as one: both arms
plateau at the identical final value (2.0), `on` simply gets there four
generations sooner and then also stops improving, exactly like `off`
eventually does. The fuller picture across all three seeds doesn't even
favor `on` consistently on arrival speed — seed 1 has `off` reach its own
shared final value (4.0) one generation *before* `on` does (gen 7 vs. gen
8). Arrival speed to a ceiling neither arm exceeds is not a meaningful
signal in either direction here; only seed 2's actual ceiling-exceeding is.

### 6.4 Fitness gain per 1k tokens

| seed | off | on | ratio |
|---|---|---|---|
| 0 | 0.0322 | 0.0128 | 2.52x worse |
| 1 | 0.0318 | 0.0129 | 2.47x worse |
| 2 | 0.0645 | 0.0426 | 1.51x worse |
| **mean** | **0.0428** | **0.0227** | **1.89x worse** |

Circles is less token-efficient on every single seed, mean 1.89x worse —
narrower than the anchor's 2.1x but not close to parity, and (§6.3) the
seed with the smallest deficit is the same seed that produced the only
fitness edge, not a different one.

### 6.5 Did the sweep's prediction hold? An 8x miss, and why it matters beyond this run

Partially. **What held**: `off` did not plateau by generation 4 the way
the anchor did — it kept improving through generation 7 (mean 2.00 -> 2.00
-> 2.00 -> 2.33 -> 2.33 -> 2.33 -> 3.00 -> 3.33), which is exactly what
`genome_length=24` was chosen to produce, and is why this comparison is
not vacuous the way the anchor's was.

**What didn't**: the deterministic sweep (classical uniform-random
`crossover()`/`mutate()`, 150-generation horizon) predicted pop=8 wouldn't
plateau until generation ~54 on average at this genome length. The real
`off` arm here — running the actual `position`/`segment` LLM operators —
plateaus around **generation 7**, roughly **8x earlier** than the free
proxy predicted. Stated plainly, not rounded off: the sweep's quantitative
timeline did not transfer to the real operator.

The likely mechanism is the caveat flagged before this run started, now
with direct evidence behind it: `PHASE2_RESULTS.md` §4.4 already
documented that `best`-style LLM operators are biased (mutation
concentrated on positions 0-1, crossover splitting near the exact
midpoint), not uniform-random like the sweep's proxy operators — so they
explore a narrower slice of the genome space per generation and converge
faster than a uniform-random search over the same landscape would. **This
generalizes past this project, the same way §4's redesign did**: a
deterministic proxy built from classical, unbiased operators can correctly
*rank* which configuration has more landscape structure to exploit (the
qualitative use this sweep was put to, and it worked — genome_length=24
was a real improvement over 18), while still being the wrong tool for
predicting *when* a biased, narrower-exploring real operator will stall.
That timeline question is answerable only by running the real operator,
not proxied from a different one, however cheap the proxy is.

### 6.6 Cost, GPU, and provenance

Total wall time: 71.1 minutes (`off`: 731.4s across 3 seeds; `on`: 3533.3s
across 3 seeds — the `on` arm alone is ~4.8x the `off` arm's cost,
consistent with 4 additional agents' propose/consult/observe calls plus
periodic central-directive/curation calls layered on top of the ordinary
fill). GPU was free at start (0 MiB, 0%) and the only `compute_apps` entry
throughout was this run's own Ollama server process — no other user's
contention to discount from the timing numbers. Total tokens: 93,468
(`off`) + 226,479 (`on`) = ~320k. Raw logs (per-seed operator-call,
diversity, and blackboard JSONL) and the aggregate summary are committed
(`f7e4ba7`) alongside the two scripts that produced them
(`sweep_diversity_config.py`, `run_circles_diversity_fitness.py`) —
finishing the run included this, per §5's retention note, not a follow-up
step.

## 7. The 15-generation follow-up: does the diversity gap compound into a fitness lead?

*Answers the question §6's follow-up closed with: `off` plateaus around
generation 7 within a 10-generation budget, leaving only ~3 generations of
runway. Does `on`'s extra diversity keep converting past that point, or
does it hit its own ceiling too? Same configuration as §6 — genome_length=24,
pop_size=8, `PROMPT_STYLE="best"`, `N_CIRCLES=2`, `AGENTS_PER_CIRCLE=2`,
3 seeds — only `n_generations` changes, 10 -> 15.
`experiments/run_circles_diversity_fitness.py` (now parameterized via
`HPGA_DIVFIT_N_GENERATIONS`), `results/raw/circles_diversity_fitness_summary_1789660105.json`.
Wall time 110.8 minutes; GPU free at start and monitored at every one of
the 6 arm-run boundaries, not just start/end (added specifically because a
second GPU consumer was expected to possibly appear on this node
mid-run) — zero contamination detected at any checkpoint.*

### 7.1 A methodological finding first: `off` is reproducible, `on` isn't — and the fix already exists, unapplied

Checked directly against the raw arrays, not assumed: `off`'s generations
0-9 in this run are byte-identical to the earlier 10-generation run's
`off`, all 3 seeds — same `best_fitness_by_gen`, same `diversity_by_gen`,
to the decimal. `on`'s are not: 2 of 3 seeds diverge from the earlier
run's `on` within the first 10 generations, despite the same nominal seed.

The mechanism is traceable, not speculative. `off`'s operator calls
(`position` mutate, `segment` crossover) almost never retry — near-0%
fallback under `best`-style, established at §4.3. `on`'s additional
circle-agent calls (`propose`/`consult`/`observe`, `hpga/circles.py`) do
retry: this run logged 17, 16, and 5 failures across the three `on`
seeds. Each retry burns extra stochastic draws that shift every
subsequent draw for that seed's run, so two executions with the "same"
seed diverge the moment a retry does or doesn't happen. `off` never hits
this because it essentially never retries; `on` does, every time.

**This is `hpga/circles.py`'s own "known open risk" note materialising
exactly where it predicted, not a new failure mode.** propose/observe use
a free-form FOLD/RATIONALE prompt and were never given the
`position`/`segment`-style affordance that took mutate and crossover from
unreliable to near-0%-fallback (§4.3 of `PHASE2_RESULTS.md`, §4 of this
document). **The fix is not unknown — it is the same fix, already
validated twice on this exact failure mode, sitting unapplied on a third
call site that turns out to need it just as much.** Practical consequence
for reading the rest of this section: the 15-generation run is a clean
extension of `off`'s trajectories only; `on`'s should be read as a
related but independently-drawn run, not literally "the same run
continued."

### 7.2 The core result, stated plainly: both arms plateau

| gen | off (mean, [min-max]) | on (mean, [min-max]) |
|---|---|---|
| 0-5 | 2.00 -> 2.33 [1-3] | 2.00 -> 2.67 [1-3] |
| 6 | 3.00 [2-4] | 2.67 [2-3] |
| 7-11 | 3.33 [2-4] | 3.33 -> 3.67 [2-5]/[3-5] |
| 12-14 | **3.67** [2-5] | **4.00** [3-5] |

Per-seed finals: off = [2.0, **5.0**, 4.0] (mean 3.67); on = [3.0, 4.0,
5.0] (mean 4.00).

Extending the budget from 10 to 15 generations bought `off` **+0.34** mean
fitness (3.33 -> 3.67) and `on` **+0.33** (3.67 -> 4.00) — statistically
indistinguishable increments. `off` reaches its own final value at
generation 6, 12, 6 across the three seeds; `on` at generation 8, 13, 7 —
`on` plateaus a little later, not never. **This is the direct answer to
the question this run was built to ask, stated as plainly as the data
supports: circles' diversity does not compound into a widening fitness
lead given more runway.** Both arms hit a ceiling within reach of the
extra generations; `on`'s ceiling is marginally higher and arrives
marginally later, and that is the entire effect five more generations
bought.

Diversity is consistent with this reading rather than contradicting it:
`on`'s diversity keeps declining from generation 9 to 14 (13.21 -> 12.21
-> 11.35 -> 11.37 -> 11.17 -> 11.35, continuing the trend already visible
in §6.2) while `off` stays flat near its own floor (~2.0-2.3) the whole
way. `on` is not sustaining diversity indefinitely, it is eroding slowly
from a much higher starting point — both quantities are moving toward
convergence, and neither ran out of room to keep separating within the
15-generation horizon actually tested.

Also worth stating plainly rather than smoothing into the aggregate: for
the first time in this project, circles produced an **outright per-seed
loss** — seed 1, off=5.0 vs on=4.0, not a tie. The picture at 15
generations is genuinely mixed (2 seeds ahead, 1 behind), not cleaner
than the 10-generation snapshot, just noisier with more information in
it.

### 7.3 The scoreboard, updated honestly

Across the six fitness comparisons this project has now run — the
anchor's four (§5: final fitness, generation-reached, and total gain all
ties; fitness-gain-per-1k-tokens a clear loss) plus one aggregate
comparison each from §6 and this section (both wins) — **circles stands
at two wins, one outright loss, three ties, at roughly 2x the token cost
throughout, including on the two comparisons it won.** That is not a case
for the architecture on this problem. It is also not a case against
LLM-based search generally, or even against circles specifically at a
different scale or on a different representation (§8) — it is a case
against continuing to tune this architecture on this problem expecting
the fitness picture to resolve in its favor.

## 8. What Phase 3 establishes, and what's next

Phase 3 establishes two robust results and one settled negative result,
and together they point at a conclusion this document previously could
only assert, not show.

The first robust result (§2, §6.2, §7.2): circles/agent architectures on
this substrate reliably produce more diverse populations than the plain
baseline, and the gap does not depend on landscape difficulty, generation
budget, or the underlying LLM operators being reliable. It held on a
plateau-limited landscape (genome_length=18, where the space is small
enough that neither arm needs strategy to cover it) and on a
trap-limited one chosen specifically to have real local optima
(genome_length=24, where pop=8 provably settles below what a larger
population reaches). It held at 10 generations and at 15. It held when
mutate/crossover were mostly falling back to the deterministic operator
and it holds now that they're fixed. Ratios ranged 1.45x-5.95x across
eight measurements total and never inverted. A candidate causal story for
part of the gap — the identical-parent feedback loop — was tested
directly and did not survive: strong (2.7x) precisely when an operator
was still broken, a small residual (1.2x) once both were fixed (§3).

The settled negative result (§7.2, §7.3): given the room to matter that
§6 was built to provide, and then given more than twice the runway that
room initially offered (§7), circles' diversity advantage did not compound
into a widening fitness lead. Both arms plateau; `on`'s plateau is
marginally higher and marginally later, worth a combined two wins across
six comparisons run so far, against one outright loss and three ties, at
roughly double the token cost throughout. This is no longer the open
question §6 left it as — it has been tested at two landscape types and two
horizons, and the answer came back the same each time.

**The real conclusion, and it is now evidence-backed rather than
asserted**: we have tested whether maintaining population diversity
converts into better search outcomes on a plateau-limited landscape and a
trap-limited one, at a short horizon and a longer one, and in every case
the diversity gap is robust while the fitness benefit isn't. On a
representation with no semantic content — the genome is a flat string of
`S`/`L`/`R`/`U`/`D` move symbols, the same point `PHASE2_RESULTS.md` §5
already made about why the LLM brings no reasoning advantage to mutation
and crossover here — maintaining diversity is apparently not the binding
constraint on search quality. If it isn't, no amount of further tuning
inside this architecture (more circles, more agents, longer
communication, a fixed propose/observe affordance) should be expected to
change that, because the mechanism being tuned was never where the
bottleneck was. **This is the argument for changing the problem rather
than continuing to tune the architecture** — moving to a representation
where sequence content is actually meaningful to reason about, rather than
a move-string where uniform-random exploration is already close to as
good as anything an LLM produces (PHASE2_RESULTS.md §5's original point,
now reinforced rather than repeated).

A methodological result worth keeping separate from either of those
(§6.5, §7.1): a free deterministic proxy correctly ranked *which*
landscape had more exploitable structure but missed *when* the real,
biased LLM operators would plateau on it by roughly 8x — and separately,
`off`'s near-zero retry rate makes it exactly reproducible run to run
while `on`'s nonzero retry rate (from a call site — propose/observe —
that was never given the affordance fix mutate and crossover already
have) makes it not. Both are the same underlying lesson from different
angles: a proxy or a baseline built from a more reliable operator behaves
more predictably than the real, less reliable one it's standing in for,
and the gap between them is only knowable by actually running the
less-reliable path.

Next:
- **Apply the position/segment-style affordance fix to propose/observe.**
  §7.1 traced `on`'s non-reproducibility to exactly the failure mode this
  fix already solved twice (§4.3). This is no longer speculative future
  work, it's a known fix with an identified, unpatched third call site.
- **Repeat the anchor configuration itself (genome_length=18) at more
  seeds.** §3's 1.2x identical-parent point and §5's original four
  fitness comparisons still rest on that single run; the multi-seed
  treatment applied at genome_length=24 (§6, §7) hasn't been applied back
  to the anchor's own configuration yet.
- **Recalibrate `CROSSOVER_MIN_DIFF` for `full`/`diff`-style crossover, or
  retire those styles.** §4's redesign only fixed `segment`; the old
  17.6%-false-rejection threshold is still the fallback gate for the two
  styles that still restate letters independently.
- **Keep the raw-log discipline the retention-gap fix (§5) doesn't
  automate.** Followed twice now (§6.6, this section) — every raw file
  these sections' numbers depend on is committed. Version control and
  backups only help if every run that produces a real number keeps doing
  that, not just these two.
  once.
