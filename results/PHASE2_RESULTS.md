# Phase 2 Results — LLM-Based Genetic Operators

*Standalone results summary, same structure as `Phase 1/results/PHASE1_RESULTS.md`.
See `Phase 2/README.md` for the operator implementation, env vars, and the
byte-identity record establishing that only `hpga/operators.py` differs from
Phase 1. Phase 3 (`hpga/circles.py`, `hpga/blackboard.py`) has its own
document, `results/PHASE3_RESULTS.md` -- §7-8 below (GPU/DIBM diagnostics,
`hpga/agents.py` communication) predate Phase 3 and are referenced by it,
not superseded, so they stay here.*

## 1. What was built and measured

Phase 2 substitutes the deterministic crossover/mutation operators in Phase
1's HPGA harness with calls to a local Ollama server (`gemma4:12b`,
Q4_K_M), selectable at runtime via `HPGA_OPERATOR_MODE` (`deterministic` or
`llm`), with everything else in the harness (`island.py`, `worker.py`,
`instrumentation.py`, dispatch, selection, elitism) byte-identical to Phase
1. Five things were measured:

1. **A paired pilot** (deterministic vs. LLM operators, `pop_size=8`,
   `n_generations=3`, `n_workers=4`) — the first look at how much of wall
   time the GA phase consumes once crossover/mutation are LLM calls.
2. **A small worker-count sweep** (`N ∈ {1,2,4,8}`, same `pop_size=8`,
   `n_generations=3`) under both operator modes, to check whether `n_workers`
   still matters once the GA phase is dominated by sequential LLM calls.
3. **A latency-vs-tokens probe**: 33 raw Ollama calls (temperature=0, so
   repeats are comparable) varying `tokens_in` and `tokens_out`
   independently, fit by least squares to make the per-token cost explicit.
4. **A diff-style operator prompt**, tried directly against the latency
   model's implication: ask the model to emit only the genome positions that
   change, instead of restating the whole genome, and measure whether
   latency drops proportionally to the output tokens saved.
5. **A reliability diagnosis of the operators themselves**, triggered by
   search-quality logs that showed the model silently doing nothing: how
   often mutate/crossover are no-ops, why, two rounds of prompt fixes, and
   what each fix costs in retries and wall time.

Deliberately run at small scale (`pop_size=8`, not Phase 1's 256) — LLM
calls are slow enough (seconds each) that a full Phase-1-scale sweep would
take hours for the same qualitative answer. Genome length is 18
(`SEQ_LENGTH=20`, two residues pinned per `hpga/config.py`).

## 2. Headline finding 1: N is architecturally cut off from wall time

Under LLM operators, the GA phase (crossover + mutation, all sequential LLM
calls) is **99.95% of wall time** (pilot: `ga_total_s=305.26s` of
`wall_time_s=305.42s` at N=1). `n_workers` only exists inside the DIS phase
(fitness dispatch to worker processes) — it has no path into the GA phase at
all, in either operator mode. With DIS phase reduced to 0.05% of wall time,
there just isn't enough of the run left for `n_workers` to act on.

**This is not a validated prediction of Xue et al.'s Eq. 3/7** (that model
describes the DIS-phase speedup ceiling, established in Phase 1, and says
nothing about the GA phase). The flat line below is a mechanical consequence
of an architecture where the LLM calls happen entirely outside the worker
pool — worker count stops mattering because it was never wired to the part
of the run that now dominates, not because a model predicted it would.

N-sweep results (fresh run, saved at `results/raw/n_sweep_transcript.txt`):

| N | wall (s), LLM | observed speedup | predicted speedup* |
|---|---|---|---|
| 1 | 305.42 | 1.000 | 1.000 |
| 2 | 290.62 | 1.051 | 1.000 |
| 4 | 284.12 | 1.075 | 1.000 |
| 8 | 283.38 | 1.078 | 1.001 |

*mechanical Amdahl bookkeeping from N=1's own `ga_total_s`/`dis_total_s`
split (`predicted_wall(N) = ga_total_s(N=1) + dis_total_s(N=1)/N`), not a
fitted or independently-derived model.

The predicted line is essentially exactly flat, as it must be given DIS is
0.05% of wall time. The observed line drifts mildly upward (1.00 → 1.08)
instead of sitting exactly at 1.00 — but `ga_total_s` itself drifted
305.3s → 283.2s across the four sequential runs despite the architecture
guaranteeing GA-phase time should be N-invariant, so this is run-to-run LLM
latency variance (see Limitations), not `n_workers` doing real work.

**Point 3 — the deterministic arm of this same sweep is not usable for
comparison.** At `pop_size=8`, deterministic-operator wall time is
0.15–0.18s total, and observed speedup is **0.83x at N=8** (i.e. a
slowdown):

| N | wall (s), deterministic | observed speedup | predicted speedup* |
|---|---|---|---|
| 1 | 0.1486 | 1.000 | 1.000 |
| 2 | 0.1423 | 1.044 | 1.998 |
| 4 | 0.1470 | 1.011 | 3.985 |
| 8 | 0.1784 | **0.833** | 7.927 |

Dispatch overhead dominates completely at this population size — 8
individuals isn't enough work to keep even 2 workers busy relative to the
fixed per-worker dispatch cost, and predicted (near-ideal) speedup diverges
wildly from observed. **Any deterministic-vs-LLM or deterministic
worker-scaling claim should use Phase 1's `pop_size=256` sweep, not this
pilot's numbers** — this table exists only as the reference/contrast point
for the LLM sweep above, not as a scaling result in its own right.

## 3. Headline finding 2: latency decomposition and the diff-style operator

Fit from 33 raw calls (temperature=0, `results/raw/latency_vs_tokens.jsonl`):

```
latency_s ≈ 1.733 + 0.00537 * tokens_in + 0.339 * tokens_out      (R² = 0.991)
```

**An output token costs ~63x an input token** (338.6ms vs. 5.4ms) — Ollama
prefills the prompt in one pass but decodes output tokens one at a time, so
the cheapest lever for cutting LLM-operator latency is emitting fewer output
tokens, not trimming prompts.

Acting on that: a diff-style prompt variant (`HPGA_LLM_PROMPT_STYLE=diff`,
`hpga/operators.py`) asks the model to emit only `<position>:<letter>` pairs
for positions that change, applied on top of a copy of the reference genome,
instead of restating all 18 letters. Paired against the existing full-genome
prompt at pilot scale (`experiments/run_diff_style_probe.py`):

| op | tokens_out (full→diff) | mean latency/call (full→diff) | requests (full→diff) | **total op time (full→diff)** | **total time change** | valid_rate (full→diff) |
|---|---|---|---|---|---|---|
| mutate | 22.0 → 9.0 (**-59%**) | 8.819s → 6.504s | 18 → 18 | 158.7s → 117.1s | **-26%** | 1.00 → 1.00 |
| crossover | 44.0 → 41.2 (-6%) | 22.494s → 19.833s | 9 → 13 | 202.4s → 257.8s | **+27%** | 1.00 → **0.62** |

"Total op time" is the sum of every request's latency, including retries —
the number that actually determines GA-phase wall time, as opposed to
"mean latency/call," which is easy to misread as a win in isolation. For
crossover, each individual request got faster (22.494s → 19.833s), but the
diff prompt needed 13 requests to complete 9 logical crossover calls (a 62%
validity rate means more than a third of first attempts were retried), so
total crossover time went **up** 27% despite every single call being
quicker. Mutate needed no retries in either style (18 requests both ways),
so its mean-latency win and total-time win agree.

**This is a real, split finding, not a uniform win:**

- **Mutation is genuinely sparse** at this config (`mutation_rate=0.05`, ~1
  of 18 positions expected to change), so a diff naturally has far fewer
  tokens than the full genome — the 59% token cut and 26% latency cut are
  real and came at no reliability cost. The latency reduction is smaller
  than the token reduction because the fixed ~1.7s per-call overhead and a
  slightly larger prompt (the diff instructions themselves cost more input
  tokens: 201 vs. 171) don't shrink — so the win is proportional to tokens
  saved, but not 1:1.
- **Crossover is not sparse under this encoding.** A recombined child
  typically differs from its reference parent across roughly half the
  genome (that's what crossover does), so listing `<position>:<letter>` for
  ~9 of 18 positions costs about as many tokens as just listing all 18
  letters — there's no structural sparsity to exploit. The added arithmetic
  (matching integer positions to a 0-indexed genome) also made the model's
  output more often unparseable, which is a real cost on its own.
- **Practical takeaway:** apply diff-style prompting per-operator based on
  whether its expected output is actually sparse relative to genome length,
  not as a blanket substitution for the full-genome format. Mutation
  qualifies here; crossover, as encoded, doesn't.

## 4. Headline finding 3: operator reliability, the affordance fix, and its cost

### 4.1 The bug: both operators were frequently silent no-ops

Search-quality logs (`results/raw/llm_operator_calls_searchquality_llm_seed{0,1,2}_*.jsonl`,
independent GA runs, not the pilot/sweep logs above) showed the LLM
operators routinely returning their input unchanged:

- **Mutate**: 150/150 calls in the seed-0 log returned a genome
  byte-identical to the input.
- **Crossover**: of the 65 crossover calls in seed 0, 63 had Parent 1
  identical to Parent 2 (population collapse from the mutate bug — the
  child matching the parent there is forced by arithmetic, not a model
  failure). Restricting to the 42 calls across all five search-quality logs
  where the parents actually differed, **39 (93%) returned both children as
  exact copies of the two parents** (one child per parent, occasionally
  swapped) — a full pass-through, not partial recombination.

One candidate explanation was ruled out directly: `_extract_labelled`
matches the literal strings `"CHILD1"`/`"MUTATED"` against the model's
*response* text only. It cannot collide with `"Parent 1"` in the *prompt* —
the parser is not the bug.

### 4.2 First fix attempt: validate and retry (works, but expensive)

Both operators originally asked the model to restate a full genome
("mutate approximately 5%..." / "recombine into two children..."). The
first fix kept that format and added reject-and-retry validation in
`hpga/operators.py`:

- **Mutate**: require the returned genome to differ from the input in
  exactly `K = max(1, round(rate * length))` positions (rate=0.05,
  length=18 → K=1); reject and retry otherwise.
- **Crossover**: require each child to differ from *both* parents in at
  least `CROSSOVER_MIN_DIFF = 2` positions (a threshold, not a strict
  inequality, to also catch a child that changes one letter just to dodge a
  `!=` check); reject and retry otherwise.

Live 10-call samples (`HPGA_LLM_MAX_RETRIES=2`, fresh random genomes each
trial, real Ollama calls):

| | mutate (full style) | crossover (full style) |
|---|---|---|
| Bad output ever returned to caller | 0/10 | 0/10 |
| Succeeded via the LLM | 3/10 | 6/10 |
| Exhausted retries → fell back to deterministic operator | **7/10 (70%)** | **4/10 (40%)** |
| Requests/call | 2.7 | 2.0 |

The validator works — no echo or no-op ever escapes — but it mostly
doesn't change what the model does. Inspecting the 7 mutate fallback calls
directly: **the model returned the byte-identical unchanged genome on all 3
attempts of every one of them**, including after a retry hint that read
*"Exactly 1 of the 18 positions must differ from the input Genome — not
approximately 1, exactly 1"* with a fresh sampling seed each attempt. That
rules out prompt ambiguity ("5%" reading as "≈1 position, maybe 0") as the
mechanism — the instruction was unambiguous and repeated, and the model
still chose to copy.

### 4.3 Second fix attempt: change the affordance, not the wording

The actual variable was the *output format*, not the instruction text. The
original diff-style mutate prompt (§3) used the same "approximately 5%"
wording as the full-style prompt but asked the model to list only the
positions that change — and that log showed 0/18 no-ops, versus 150/150 for
the full-restate format on the same instruction. When the model has to
restate the whole genome, copying it is the lowest-effort valid answer.
When it can only name changes, copying isn't an expressible answer at all.

Two new operator formats apply this directly, both selectable via
`HPGA_LLM_PROMPT_STYLE` in `hpga/operators.py`:

- **`position` (mutate)**: emit `POSITION: <n>, NEW: <letter>` line(s)
  only, with `NEW` constrained to differ from the genome's current letter
  at that position — parsed and applied in code, no genome restated.
- **`segment` (crossover)**: emit a partition of the genome into contiguous
  segments with a source-parent label per segment (`SEGMENTS:
  <start>-<end>:<parent>, ...`), applied in code; CHILD2 is the
  parent-complement of CHILD1. No letters generated by the model at all —
  it only chooses *how* to split and *which* parent supplies each piece.

Live samples (50 calls each: an initial 10-call check, then a 40-call
distribution run):

| | `position` mutate | `segment` crossover |
|---|---|---|
| No-op / echo ever returned | **0/50** | **0/50** |
| Fell back to deterministic | **0/50** | **0/50** |
| Retries | 4/54 requests | 0/50 requests |
| Mean latency/request | 8.5–9.2s | ~14.0s |

Compliance is perfect: every one of 100 combined calls across the two new
formats produced a valid, non-echo result on essentially the first attempt.
This confirms the affordance framing — not vague wording (the "exactly K"
instruction was already unambiguous before this fix and didn't help) and
not blanket model refusal (the model isn't incapable of following the
instruction, §4.2's 3/10 and 6/10 LLM successes show that) — the format
itself was what made copying available as an answer.

### 4.4 The catch: reliable is not the same as useful

Neither fix produces what a genetic operator actually needs — variation.
Quantified over the 40-call distribution runs:

- **`position` mutate**: chosen positions are extremely non-uniform.
  Position 1 was chosen 20/40 times (50%), position 0 twelve times (30%) —
  **80% of all mutations land on the genome's first two positions**, and
  only 7 of the 18 positions were ever touched at all. Chi-square against a
  uniform distribution over 18 positions (df=17): **211.10** — not sampling
  noise at that scale.
- **`segment` crossover**: every one of 40 calls used exactly 2 segments
  (never 3+), and the cut point landed at position 9 — the exact genome
  midpoint (length=18) — in 36/40 calls, and position 8 in the remaining 4.
  In substance, **40/40 calls produced a single-point crossover at or one
  position off the exact midpoint.** This is materially the point-based
  design considered and rejected earlier in this diagnosis for reducing the
  LLM to picking an integer for `random.randint()` to consume — except here
  the model isn't even varying the point most of the time.

So: both affordance fixes eliminate the compliance failure completely, but
what replaces "copy the input" is not exploration — it's the next
cheapest valid answer the format admits (first position; midpoint split).
"Reliable" and "behaves like a useful GA operator" are separate claims, and
only the first one holds. Any future search-quality comparison that uses
these formats needs to control for this rather than assume LLM-mutate ≈
uniform-random mutate with added latency.

### 4.5 The fallback-cost irony

Put together with §4.2: the validate-and-retry fixes that *do* keep the
free-form (letter-generating) prompt format run at 70% (mutate) / 40%
(crossover) total-fallback rates. In roughly 2 of every 3 mutate calls and
2 of every 5 crossover calls under that design, the "LLM operator" arm
spends 2-3 full-latency LLM requests (mutate: ~2.7 × 12.7s ≈ 34s;
crossover: ~2.0 × 21.0s ≈ 42s, worst case 3× single-request latency) only
to then execute the ordinary deterministic operator — `random.choice` /
`random.randint`, microseconds — anyway. §2 measured the GA phase at
99.95% of wall time under the *original, uncorrected* operators; layering
reject-and-retry correctness on top of the same free-form format would
*increase* that number, not reduce it.

It's worth being precise about which problem the §4.3 affordance formats
actually fix, because it is not this one. `position` mutate ran at
8.5-9.2s/request versus ~8.8s for the original full-style prompt — no
faster per call — and `segment` crossover ran ~14s/request, not a
token-count win the way §3's diff-style mutate prompt was. What the
affordance formats remove is the retries and fallback themselves (§4.2's
70%/40% rates go to 0%), which is a genuine reliability win. It does
nothing for the serial-fraction number: the GA phase is still 100%
sequential LLM calls under every variant tested in this document, so §2's
99.95% stands regardless of prompt format. Reliability and serial fraction
are independent axes here — fixing the first does not touch the second.

## 5. Limitations

- **On this problem, the LLM brings no semantic advantage — §4.4 measured
  this directly, not just argued it.** The `segment` crossover format
  places no constraint on segment count or cut placement; the model is free
  to use 2, 3, or a dozen segments anywhere it likes. Given that freedom, it
  chose exactly 2 segments in 40/40 sampled calls and landed within one
  position of the exact genome midpoint in all 40 (36/40 at the precise
  midpoint). Nobody encoded single-point crossover into the prompt or the
  parser — the model converged there on its own, at ~14s per call, to
  reproduce what `hpga/operators.py`'s deterministic single-point
  `crossover()` already does in microseconds. That is direct evidence, not
  an inference from the encoding: the genome is a flat string of S/L/R
  symbols with no structure for a language model to reason about, so a
  uniform-random symbol swap (`rng.choice`, microseconds) is exactly as
  good a mutation as anything the model produces at ~9s of latency, and the
  model's own unconstrained choice on crossover reproduces the
  deterministic operator's behavior rather than discovering anything beyond
  it. The LLM is in
  this harness for a different reason entirely — to put `T_calc`/
  `T_turnaround`-scale latency into a large, realistic regime and measure
  what that does to the bottleneck model (the serial-fraction collapse in
  §2), not to improve search quality, and §2/§3's results should be read
  that way. The semantic case for LLM-based genetic operators — reasoning
  about structure a classical operator can't easily encode — applies to
  genomes with actual semantic content (programs, prompts), not to lattice
  move strings; Phase 3 and beyond, if they continue using LLM operators on
  this same HP-lattice representation, should be read with this distinction
  in mind rather than as evidence that LLM operators improve GA search here.
- **Session-level LLM latency variance**, as already flagged in Phase 1 for
  wall-clock timing generally: `ga_total_s` drifted ~7% (305.3s → 283.2s)
  across four sequential, architecturally-identical LLM sweep runs. The
  mild upward drift in observed N-sweep speedup (§2) is very likely this,
  not a real effect of `n_workers`, but that hasn't been isolated from a
  genuine (if tiny) `n_workers` effect on the sub-0.05%-of-wall-time DIS
  phase.
- **Diff-style crossover's 0.62 valid rate is one small paired run** (13
  vs. 9 requests) — not enough samples for a stable estimate of the failure
  rate, though the underlying mechanism (crossover output isn't sparse
  under a per-position diff) is structural, not a sampling artifact, and
  should reproduce.
- **Everything here is at `pop_size=8`, `n_generations=3`.** That scale was
  chosen deliberately to keep LLM wall-clock time tractable for a
  qualitative check (serial-fraction collapse, diff-style token savings);
  absolute magnitudes, and possibly the crossover retry rate, may shift at
  Phase-1 scale.
- **Single model, single local deployment.** `gemma4:12b` (Q4_K_M) on one
  Ollama instance — the 338.6ms/output-token constant is a property of this
  deployment (quantization, hardware, `keep_alive`/batching config), not a
  universal LLM-inference constant, and the latency-vs-tokens fit was taken
  at temperature=0 while the operators themselves default to temperature=0.7.
- **§4's reliability and bias numbers are single-model, moderate-sample
  live measurements** (n=10-65 per condition, n=40 for the two distribution
  studies) against the same `gemma4:12b` deployment above, at
  temperature=0.7. The qualitative pattern (restating enables copying;
  freedom-to-choose collapses to the cheapest valid choice) is a claim
  about this model's behavior under these exact prompt formats, not a
  universal LLM property — a different model or a reworded `position`/
  `segment` prompt (e.g. explicitly banning the two most common outputs)
  could plausibly shift the specific numbers even if the underlying
  mechanism (lowest-effort-valid-answer) reproduces.

## 6. What Phase 2 establishes, and what's next

Phase 2 establishes two independent things about running LLM genetic
operators on this HP-lattice representation, and they compound rather than
cancel.

First (§2), swapping in LLM operators doesn't just make each generation
slower — it restructures where wall time goes: serial fraction jumps from
Phase 1's 8.9% (deterministic) to 99.95% (LLM) at this config, which
architecturally strands the worker-count lever this whole harness was built
to study. The latency model built to explain that cost (§3, output tokens
dominate input tokens by ~63x) pointed at a design change — a diff-style
prompt — validated for mutation (a real 26% latency cut) while cleanly
failing for crossover, for a structural reason (lack of output sparsity)
rather than an implementation bug.

Second (§4), and found only by inspecting search-quality logs rather than
the pilot/sweep runs above: on this representation, the LLM operators are
*unreliable*, not just slow. At the original free-form prompt, mutate was a
no-op 150/150 times in one full run and crossover echoed a parent verbatim
in 93% of calls with genuinely differing parents. Fixing that by validating
output and retrying works, but pays for itself in a genuinely ironic way:
70% of mutate calls and 40% of crossover calls burn 2-3 full-latency LLM
requests (10-60s) only to fall back to the microsecond-scale deterministic
operator anyway — meaning under that design, the "LLM arm" of the
comparison is mostly *running the deterministic arm at LLM latency*. The
fix that actually works — changing what the model is asked to produce
(`position` for mutate, `segment` for crossover) instead of how the
instruction is worded — gets compliance to 100% with no retries. But it
trades the no-op/echo failure for a different one: the model's "choice",
once copying is foreclosed, collapses to the cheapest valid alternative
(mutate position 0-1 in 80% of calls; crossover splits at the exact
midpoint in 40/40 calls) rather than exploring the space the way
`rng.choice`/`rng.randint` do by construction. Combined with §4's own
limitations bullet, that closes the loop opened in the first Limitations
bullet above: the LLM adds no semantic value on this genome, and here it
doesn't even reliably add *variation* without deliberate format engineering
— it adds cost, and untuned, it adds unreliability on top of the cost.

Next:
- **Adopt `position`/`segment` as the default LLM operator formats** going
  forward (§4.3-4.4) — they're the only variants tested that are both
  100%-compliant and don't pay the 40-70% fallback tax of §4.2's
  validate-and-retry fix on the free-form formats. Any run that reports
  them as "LLM mutation/crossover" without also reporting the position/
  segment-choice skew (§4.4) is overstating what the operator does.
- **If search quality is ever compared LLM-vs-deterministic on this
  representation, control for §4.4's non-uniformity first** — an operator
  that mutates positions 0-1 in 80% of calls is not comparable to
  `rng.choice`, and any quality difference observed could be an artifact of
  that skew rather than of using an LLM at all.
- Re-run the N-sweep and diff-style probe at Phase 1's `pop_size=256` scale,
  budget permitting, to confirm the serial-fraction collapse and the
  mutation/crossover split both hold at scale and aren't small-`pop_size`
  artifacts. Re-running §4's reliability/bias measurements at that scale
  would also show whether the fallback rates and position/segment skew
  are stable or config-dependent.
- Refit the latency-vs-tokens model at temperature=0.7 (the operators'
  actual default) rather than temperature=0 (the probe's determinism
  choice), to check the per-token constants transfer to the sampling
  regime actually used.

## 7. GPU residency and the DIBM parallel-dispatch limit

*A later diagnostic, prompted by discovering mid-session that this machine's
GPU (GTX 1650 Ti, 4GB VRAM) could not hold `gemma4:12b` (7.6GB, Q4_K_M) --
every number in Sections 1-6 above was measured with that model running
mostly on CPU. This section asks two questions the paper's DIBM model
(Eq. 11: capacity and speedup scale by P, the number of borrowed injection
channels) makes testable: whether §3's 338.6ms/output-token constant was a
property of LLM decoding or of CPU-bound compute, and whether issuing
genetic-operator calls concurrently from the GA-phase master scales the way
Eq. 11 predicts. Scripts: `experiments/probe_latency_vs_tokens.py` and
`experiments/probe_ollama_concurrency.py` (both now take model from
`HPGA_PROBE_MODEL`), `experiments/probe_operator_compliance.py`,
`experiments/run_ga_dispatch_sweep.py`. Raw logs under
`results/raw/*_llama3.2_1b*` and `results/raw/llm_operator_calls_phase2_dispatch_*`.*

**Model-geometry disclosure, added at merge time:** every measurement in this section -- the CPU/GPU
latency refit (§7.1), the concurrency probe, the operator-compliance table,
and the DISPATCH_P sweep (§7.3) -- was run against the **pre-migration 2D
model, 3-symbol (`S`/`L`/`R`) genomes**. `hpga/hp_model.py` was migrated to
the 3D simple-cubic model (5-symbol genomes, §"3D migration" in `README.md`)
*after* this section's runs, and none of §7's runs were repeated on the 3D
model. That does not compromise what's measured here: these are wall-clock
latency, GPU-residency, and dispatch-concurrency results, none of which are
sensitive to lattice dimensionality or alphabet size -- the same model
serving the same token counts takes the same time regardless of what the
tokens decode to. But the numbers below should not be read as characterizing
the current 3D-genome operator path, only as characterizing this Ollama
deployment's timing behavior at the point they were measured.

### 7.0 Model substitution and its own disqualification

The natural fix -- pull a model that fits in 4GB -- ran into a premise error
worth recording: `gemma4:e2b`'s "e2b" names *effective* active parameters
(a sparse/matryoshka naming convention), not on-disk size; every `e2b`
variant checked was 4.3-10GB, none of which fit either. `llama3.2:3b` (2.0GB)
came closer but still split 80%/20% GPU/CPU (2.9GB runtime footprint against
~3.9GB usable VRAM leaves too little headroom for the KV cache).
`llama3.2:1b` (1.3GB on disk, 1.5GB runtime) is the first model that loaded
**100% GPU**, with ~1GB of VRAM free afterward.

That model is measured here purely as a **timing instrument**, not as a
genetic operator. Directly measured (§7.1): at this size, `llama3.2:1b`
fails the mutate/crossover format contracts far more often than
`gemma4:12b` did, including on the `position`/`segment` affordance formats
that were 100%-compliant at 12b (§4.3-4.4). **Every wall-clock number in
§7.2-7.4 below was generated with this model issuing and having its
responses timed; none of them should be read as saying anything about
`llama3.2:1b`'s competence as an operator, or about the affordance fix's
general validity beyond the size at which it was tested.**

### 7.1 Stage 0: CPU vs. GPU decomposes the latency constants

Refitting `probe_latency_vs_tokens.py` on the GPU-resident model (33 calls,
temperature=0, `results/raw/latency_vs_tokens_llama3.2_1b.jsonl`):

```
gemma4:12b/CPU:   latency_s ~= 1.733 + 0.00537*tokens_in + 0.339*tokens_out    (R^2=0.991)
llama3.2:1b/GPU:  latency_s ~= 0.013 + 0.00034*tokens_in + 0.01417*tokens_out  (R^2=0.990)
```

All three constants collapse together: fixed overhead drops ~133x
(1.733s -> 0.013s), the output-token cost drops ~24x (338.6ms -> 14.2ms),
the input-token cost drops ~16x (5.4ms -> 0.34ms). This **upgrades §5's
"a property of this deployment" hedge to a demonstrated fact**: the
338.6ms/output-token constant was CPU decode cost, not something intrinsic
to LLM inference.

The concurrency probe (`probe_ollama_concurrency.py`, K in {1,2,4,8}, default
Ollama config) tells the same story at the request-batching level:

| K | gemma4:12b/CPU | llama3.2:1b/GPU (default config) |
|---|---|---|
| 1 | 1.00 | 1.00 |
| 2 | 1.49 | 1.96 |
| 4 | 1.96 | 3.75 |
| 8 | 2.28 | **6.98** |

Near-linear on the GPU-resident model vs. a flat ceiling on 12b/CPU: the
2.3x ceiling documented in this section's predecessor probe was CPU
contention, not an Ollama batching limit.

Operator compliance (`probe_operator_compliance.py`, N=20 live calls/condition,
genome length 18, seed=0) moves the opposite direction:

| condition | fallback rate, `llama3.2:1b` | fallback rate, `gemma4:12b` (§4.2-4.4) |
|---|---|---|
| mutate/full | **100%** (20/20) | 70% |
| mutate/position | **100%** (20/20) | 0% |
| crossover/full | **95%** (19/20) | 40% |
| crossover/segment | **20%** (4/20) | 0% |

The affordance fix (§4.3: change what the model is asked to produce, not how
it's worded) that eliminated 12b's no-op problem does not transfer down --
`mutate/position` goes from 0% to 100% fallback. The failure mode is
different, too: 12b's problem was copying the input (the cheapest *valid*
answer under the free-form format); 1b mostly cannot produce a valid answer
at all -- observed responses include wrong-length restatements (13-16
letters returned for an 18-position genome) and, in one case, a response
that substituted `T`/`G` for the `S`/`L`/`R` move alphabet entirely,
apparently conflating "genetic algorithm" with DNA bases. **This qualifies
§4.3-4.4 rather than contradicting it: the affordance fix is not a
universal prompt-design principle, it works where the model is capable
enough to exploit the format, and below some capability threshold the same
constraint becomes just another way to fail.**

### 7.2 Stage 1: the memory wall, and a contaminated-baseline catch

Ollama's default `OLLAMA_NUM_PARALLEL` on this install is 1 (confirmed from
the server startup log), yet the plain concurrency probe above already
scaled past 1x -- the per-model runtime auto-selects more slots than that
global default at load time. Raising it explicitly (`OLLAMA_NUM_PARALLEL=8`,
daemon restarted manually to pick it up) changed `llama3.2:1b`'s residency
from 100% GPU / 1.5GB to **85%/15% GPU/CPU / 2.8GB**: reserving KV-cache
capacity for 8 parallel slots at model-load time doesn't fit this card's
remaining ~1GB of headroom, so part of the model spills to CPU. The
concurrency probe re-run under this config barely moved (K=8: 6.61x vs.
6.98x) -- for tiny 16-token requests, the spillover cost request-level
throughput almost nothing.

**It cost something else, and this is the catch worth stating plainly.**
The first version of the Stage 2 sweep (§7.3) was run with
`OLLAMA_NUM_PARALLEL=8` still set. Its P=1 control measured
`ga_total_s=79.96s`. After restarting the daemon back to the default config
(confirmed 100% GPU residency) and re-running the *identical* P=1
configuration, the same measurement came back at **35.52s -- 2.25x faster,
from restarting the daemon alone, with nothing else about the run changed.**
The mechanism: `OLLAMA_NUM_PARALLEL` reservation happens once, at model-load
time, independent of how many requests actually arrive concurrently -- so
the CPU-spillover penalty it creates applies to *every* request the model
serves afterward, including a fully sequential one. This is a **static,
load-time capacity effect**, categorically different from the **dynamic,
request-time queuing effect** that concurrent dispatch (§7.3) is actually
trying to measure -- and it had been silently inflating every P=1 baseline,
and therefore every speedup ratio, in the first version of that sweep. All
§7.3 numbers below are from the corrected, default-config runs.

### 7.3 Stage 2: parallel operator dispatch (the DIBM analogue), repeated

Implemented `HPGA_GA_DISPATCH_P` in `hpga/operators.py`'s `next_generation()`.
At P=1 (default) the loop is unchanged. At P>1, in LLM mode only, parent
selection for all of a generation's reproduction units still runs
sequentially first (it mutates the shared RNG), then each unit's
crossover+mutate+mutate chain runs as an independent job -- its own
seeded `random.Random`, no shared mutable state -- across a
`ThreadPoolExecutor(max_workers=P)`: the master borrows P injection channels
instead of serialising through one. Shared stats counters and the JSONL
call log are now lock-guarded, since P>1 mutates them from multiple threads;
this doesn't affect P=1 timing. `island.py`/`worker.py`/`instrumentation.py`
untouched.

Every point below is **mean +/- sd over 3 repeats** (`--repeats 3`), each
repeat a different run seed (different population and operator-call RNG
stream, same target sequence) -- the first version of this sweep reported
single-run points, and one of them (§7.3.1) turned out to be noise.

**pop_size=8, n_generations=3, elitism=2 -> 3 reproduction units/generation:**

| P | ga_total_s (mean +/- sd) | observed speedup | ideal (=P) |
|---|---|---|---|
| 1 | 26.62 +/- 3.71 | 1.00 | 1.00 |
| 2 | 23.07 +/- 1.11 | 1.15 | 2.00 |
| 4 | 22.53 +/- 1.48 | 1.18 | 4.00 |
| 8 | 21.71 +/- 0.83 | 1.23 | 8.00 |

**pop_size=32, n_generations=1, elitism=2 -> 15 reproduction units/generation:**

| P | ga_total_s (mean +/- sd) | observed speedup | ideal (=P) |
|---|---|---|---|
| 1 | 46.58 +/- 1.14 | 1.00 | 1.00 |
| 2 | 46.41 +/- 3.18 | 1.00 | 2.00 |
| 4 | 44.88 +/- 5.88 | 1.04 | 4.00 |
| 8 | 42.98 +/- 3.57 | 1.08 | 8.00 |

#### 7.3.1 The single-run "regression" did not reproduce

A first pass at the pop_size=8 sweep, one run per P, showed P=4 and P=8
*slower* than P=1 (0.80x, 0.81x) -- read at the time as concurrent dispatch
being actively harmful, with a competing explanation half-considered (5 idle
threads chasing 3 real jobs at P=8, thread-pool overhead dominating a small
total). With 3 repeats, that specific reading does not survive: the sd bands
at every P overlap, and the single run that produced the "0.80x" result had
drawn a P=1 time of 35.52s -- about 2.4 sd above the repeated P=1 mean of
26.62 +/- 3.71s, i.e. an outlier low-latency day for P=1, not a demonstrated
harm from P=4/8. Phase 1 already documented ~6% session-level latency drift
with *no* concurrent load at all; three points is not enough to separate
that from a real effect, which is exactly why this correction matters more
than the number it replaces.

#### 7.3.2 pop_size=32 rules out the structural fan-out ceiling as the explanation

At pop_size=8, only 3 reproduction units exist per generation, so P>3 has no
additional independent work to hand to -- that alone predicts a plateau
near 3x once P>=3, and doesn't by itself distinguish "P saturates from GPU
contention" from "P saturates because there were only 3 jobs." Re-running
at pop_size=32 (15 units/generation, comfortably above every P tested)
produces the **same flat pattern** (1.00 -> 1.00 -> 1.04 -> 1.08) -- with
abundant real concurrent work available at every P, parallel dispatch still
does not scale. The earlier saturation at pop_size=8 was not primarily the
structural ceiling; it is dominated by contention for the shared substrate,
confirmed by direct measurement rather than inferred from the arithmetic.

Read together, both tables also make the honest null case: at n=3 repeats,
neither config shows a P-effect clearly separable from noise. There is a
small, consistent, monotonic upward trend with P in both tables (never a
reproducible regression), but every adjacent-P difference is within about
1 sd of the next, a long way from anything resembling Eq. 11's ideal=P line.

### 7.4 What this establishes: Eq. 11 and two separate substrate limits

**Eq. 11 predicts capacity and speedup scale by P. On this substrate, at
every P and every population size tested, observed speedup sits in a
1.00x-1.23x band against an ideal of 2x-8x** -- not "saturates below P," a
near-total absence of the predicted scaling. Two mechanisms are responsible,
and they are worth keeping separate because only one of them is inside
Eq. 11's own domain:

1. **Dynamic contention** (§7.3): concurrent requests share one GPU's
   compute and memory bandwidth rather than running on independent
   hardware. This is the regime a queuing/channel model like Eq. 11 is built
   to describe -- and even so, on this hardware, contention alone flattens
   the P-curve almost entirely.
2. **Static memory admission wall** (§7.2): raising the *requested*
   parallelism reserves KV-cache capacity for all P slots at model-load
   time, before any request arrives. On a 4GB card that reservation alone
   can exceed free VRAM and force part of the model onto CPU -- a penalty
   that then applies to every subsequent request, concurrent or not. This
   is a load-time capacity effect, not a runtime queuing effect, and **Eq. 11
   has no term for it**: a NoC channel model has nothing resembling it,
   because dedicating a channel to a flow doesn't touch a shared, finite
   on-die resource that every flow's own processing element also needs
   just to run at all.

Both are DIBM results. Eq. 11's capacity/speedup-scales-by-P prediction
assumes the borrowed channels are independent, as they are in a NoC; this
hardware tests what happens when they are borrowed from one shared,
memory-constrained GPU instead, and the answer is that the assumption fails
in two structurally different ways at once. **If this is Phase 4's first
question, it is answered here** -- on `llama3.2:1b`, used throughout this
section strictly as a request-issue-and-time instrument, with no operator-
validity claim attached (§7.0, §7.1).

### 7.5 Limitations specific to this section

- **P=1 and P>1 take different code paths inside `next_generation()`**: P=1
  falls through to the original sequential loop (consuming the shared `rng`
  directly, call by call), while P>1 pre-draws all of a generation's parent
  pairs and per-job seeds up front, then runs jobs against independent
  `random.Random` instances. Both are deterministic and produce the same
  *kind* of workload, but they are not the same code, so a P=1-vs-P=2
  comparison is not quite as clean as a P=2-vs-P=4 one. Not believed to
  explain §7.3's results (the flat pattern holds P=2-through-P=8, where the
  code path is identical throughout), but worth flagging rather than
  asserting away.
- **n=3 repeats** is enough to catch the §7.3.1 outlier and to see that the
  pop_size=8 and pop_size=32 tables broadly agree, but not enough for a
  formal significance test against the small P=1-to-P=8 trend that remains
  in both tables. That trend may be real (mild) or may still be residual
  session-level drift; more repeats would separate them.
- **`llama3.2:1b`'s ~100% mutate-fallback rate (§7.0, §7.1)** means almost
  every timed "operator call" in §7.3 is in practice 3 wasted LLM requests
  followed by the deterministic fallback. This is fine for the timing
  question this section asks (calls still had to be issued, sent over the
  network, and waited on, which is what contends for the substrate), but it
  means §7.3's absolute `ga_total_s` values are not comparable to what a
  compliant operator would cost at this model size, only to each other
  across P.
- **Single hardware, single small-VRAM configuration.** The specific
  finding that `OLLAMA_NUM_PARALLEL` hits a memory wall before a compute one
  is a property of this 4GB card; a card with more headroom would show the
  dynamic-contention mechanism (§7.4, item 1) in isolation, without item 2
  masking or compounding it, and that comparison was out of scope for this
  hardware.

### 7.6 gemma4:12b smoke check on the 3D alphabet

*Numbered 7.6, not 7.5 as first planned, because §7.5 above ("Limitations
specific to this section") was already written when this check was added at
merge time.*

Everything in §7.0-7.4 was run on the pre-migration 2D, 3-symbol model (see
the disclosure at the top of §7). `hpga/operators.py`'s LLM operator path
was subsequently widened to the 3D model's 5-symbol alphabet
(S/L/R/U/D) as part of that merge, and verified not to crash
(`_genome_to_str` previously `KeyError`'d on any UP/DOWN move) with a
2-generation smoke test on `llama3.2:1b` -- but `llama3.2:1b` fell back to
the deterministic operator on 32 of 33 calls in that test, so it could only
prove the parser accepts a valid 5-symbol response when one arrives, not
that a real model produces one reliably. This section asks the follow-up
question -- **does a model with actually-good 2D format compliance
(§4.3-4.4) still have it on the 3D alphabet?** -- in two passes: an initial
n=3-per-condition smoke check across all four (operator, style)
conditions, then a properly-powered n=20 follow-up on the two conditions
the smoke check found working, matching §4.2-4.4's 2D methodology.

#### 7.6.1 n=20 result: mutate/position and crossover/segment

(`HPGA_LLM_MODEL=gemma4:12b`, n=20 calls/condition, genome length=18,
seed=1 -- a fresh seed, not an extension of the n=3 run below; raw:
`results/raw/operator_compliance_gemma4_12b_n20.json`)

| condition | fallback_rate (n=20) | requests/call | mean_latency_s |
|---|---|---|---|
| mutate/position | **0.00 (0/20)** | 1.00 | 9.3 |
| crossover/segment | **0.00 (0/20)** | 0.95 | 14.6 |

**Fallback stayed at 0% for both conditions at n=20, same as n=3 -- this is
stated plainly as an observed result, not assumed as the expected outcome
of a larger sample.** It could have gone the other way (§7.6.2's other two
conditions were at 67% fallback on the same model), and didn't.

**The position-collapse question from §7.6.2 is answered, but not the way
a strict reading of "20/20" would have put it.** `mutate/position` did
not collapse onto a single position at n=20: `position_counts={"0": 3,
"1": 15, "4": 1, "17": 1}` -- 4 of the 18 possible positions were used at
least once, not 1. That rules out an absolute per-call collapse (the
model is not mechanically incapable of naming a different position). But
15/20 calls (75%) still landed on position 1 alone, against a uniform
prior of 1/18 (~5.6%) per position -- a strong, reproducible bias toward
one position, just not a total one. **This promotes the observation from
"3/3 at n=3, could be noise" to a confirmed, quantified bias (75% on one
position, 14/18 positions never chosen once in 20 calls) -- not to a
confirmed total collapse, which the data does not support.** `crossover/
segment`'s one `echo_outputs` (1/20) lines up exactly with its one
`n_skipped_no_op` (the rate-gate skip, not a model failure, per the §7.6.2
note below) -- at n=20 that condition shows no signal of genuine model
echoing at all.

#### 7.6.2 n=3 smoke check across all four conditions (context, not confidence)

`experiments/probe_operator_compliance.py` (already in the tree from the
merge) fires live calls through the real `_llm_mutate`/`_llm_crossover`
path and reports fallback rate per (operator, style) condition -- but its
own `random_genome()` hardcoded the 2D 3-symbol alphabet (`rng.choice((0,
1, 2))`), a leftover from when it was written against the 2D model. Fixed
to draw from `hpga.hp_model.MOVES` (5 symbols) before running, so this
check actually exercised the 3D alphabet rather than silently re-testing
the 2D one under a "3D" label. **This pass is a smoke-level check across
all four conditions, not a 3D replication of §4.2-4.4's n=20 methodology
-- it ran n=3 per condition.** Its numbers below are not 3D compliance
rates comparable to §4.2-4.4's 2D ones; n=3 is not powered to support
that, and the 2D compliance rates established there should not be assumed
to carry over to 3D. Its value is breadth (all four conditions, not just
the two that turned out to work) at low cost, not confidence.

(`HPGA_LLM_MODEL=gemma4:12b`, n=3 calls/condition, genome length=18,
seed=0; raw: `results/raw/operator_compliance_gemma4_12b.json`,
`results/raw/llm_operator_calls_1788628548_16024.jsonl`):

| condition | fallback_rate (n=3) | mean_latency_s | note |
|---|---|---|---|
| mutate/full | 0.67 (2/3) | 16.5 | 2/3 zero-diff (copied input) -- same failure mode as §4.1 |
| mutate/position | 0.00 (0/3) | 7.8 | all 3 on position 1 -- see §7.6.1 for the n=20 follow-up |
| crossover/full | 0.67 (2/3) | 25.7 | |
| crossover/segment | 0.00 (0/3) | 14.6 | |

One observation from this pass that §7.6.1's n=20 run didn't re-test
(neither `mutate/full` nor `crossover/full` was re-run at n=20, since they
were the two conditions the smoke check found *not* working):
**`echo_outputs` overstates model failure at face value.** Both crossover
conditions logged `n_skipped_no_op: 1` -- one of the 3 calls in each
condition never reached the LLM at all, because `crossover_rate`'s own
rate gate (independent of model behavior) chose not to recombine that
pair and returned the parents unchanged by design. That no-op is
indistinguishable from a genuine echo in the `echo_outputs` count (2/3 for
`crossover/full`, 1/3 for `crossover/segment`), so at this sample size a
meaningful fraction of "echoes" are intentional skips, not model copying
-- confirmed again at n=20 in §7.6.1 above, where the sole echo exactly
matches the sole skip.

Latency (7.8s-25.7s mean per call at n=3; 9.3s/14.6s at n=20) confirms
§7.0-7.1's CPU-bound cost applies here too, which is why the smoke pass
ran at n=3 across all four conditions rather than n=20 across all four --
`mutate/full` and `crossover/full`'s poor compliance means a full n=20
matrix was not worth the wall-clock cost once the smoke check showed
which two conditions were actually candidates for it.

## 8. Agent communication: does it pay for itself?

New architecture, not a fix: `hpga/agents.py` introduces `A` agents per
island, each a persistent identity (a stable id and a role, "explore" or
"refine", read once and fixed for the run -- no "recombine" role,
`probe_prompt_diversity.py` measured 0/10 outputs mechanically consistent
with combining two parents, so crossover stays deterministic in the
master, untouched by this feature). Agents are additive: `next_generation()`
runs its ordinary tournament-select/crossover/mutate fill to `pop_size`
exactly as the agents-off baseline does, then `A` agent-produced genomes
are appended on top -- an earlier version of this module displaced
non-elite slots instead, which meant an agents-on run made fewer top-level
LLM calls than the baseline, confounding any fitness comparison by call
budget alone; additive removes that by construction, at the cost of a
population that grows from `pop_size` to `pop_size + A` after the first
generation and stays there. Communication, switchable
(`HPGA_AGENTS_ENABLED`, off by default) and off by default, is
central-node and unaggregated: every `HPGA_COMM_INTERVAL` generations each
agent's current fold is "sent" to the master and "forwarded" individually
to every other agent (`A + A*(A-1)` messages per round, not one broadcast
blob), so a later peer-to-peer topology only removes the master hop,
doesn't restructure the content. `hpga/island.py`, `worker.py`,
`instrumentation.py`, and `config.py` are untouched -- see that module's
docstring for how (a generation counter ticked inside `next_generation()`,
an `HPGA_ISLAND_ID` env var, `_dispatch_and_collect` already sizing off
`len(population)` rather than a hardcoded `pop_size`).

This section reports a 3-arm run built specifically to separate two things
a 2-arm (off/on) smoke run at genome_length=8 could not: whether
communication or the explore/refine roles themselves were responsible for
a diversity effect seen in that first run, and whether fitness could move
at all at a slightly harder scale (it hadn't at genome_length=8 -- both
arms sat at a flat 1.0, plausibly already at the ceiling for so trivial an
instance). Setup: `gemma4:12b`, `pop_size=5`, `elitism=1`, `A=2` (1
explore + 1 refine), `n_generations=6`, sequence
`make_timing_sequence(length=14, seed=1)` = `HHPHPPPPHHPHPP` (genome_length=12,
a random string with no known optimum -- appropriate for a relative
across-arm comparison, not benchmark validation). Three arms: **off**
(agents disabled, today's baseline, unchanged), **nocomm** (agents
enabled, `HPGA_COMM_INTERVAL=100`, past `n_generations` so it never
fires), **comm** (agents enabled, `HPGA_COMM_INTERVAL=2`, fires at
generations 2 and 4). Raw: `results/raw/llm_operator_calls_agents3_{off,nocomm,comm}_1788635605.jsonl`,
`results/raw/diversity_agents3_{off,nocomm,comm}_1788635605.jsonl`,
`results/raw/agents_3arm_summary_1788635605.json`.

### 8.1 Communication does not pay for itself

`nocomm` and `comm` are the matched-budget comparison -- both make the
same 47 top-level LLM calls, so any difference between them is
attributable to communication, not to a call-count confound the way
comparing either against `off` (36 calls) would be. Lead with the control
that makes the token cost trustworthy:

| arm | comm-gen tokens_in (mean) | non-comm-gen tokens_in (mean) |
|---|---|---|
| nocomm | 202.5 | 202.5 |
| comm | 241.5 | 202.5 |

`nocomm`'s two numbers are **exactly equal** -- generation number alone
does not change prompt size in this design, only a comm round firing
does. That control is what makes `comm`'s **+39.0 tokens on comm
generations** attributable specifically to the forwarded peer fold, not
to some other confound riding along with generation number. (Consistent
in direction and rough size with the first, genome_length=8 smoke run's
measured +35 -- see §7.6's sibling note on communication cost.)

Against that real, measured cost, here is what it bought:

```
nocomm: best_fitness_by_gen = [1.0, 1.0, 2.0, 2.0, 2.0, 2.0]   total_tokens=29,997
comm:   best_fitness_by_gen = [1.0, 1.0, 2.0, 2.0, 2.0, 2.0]   total_tokens=30,153
```

Identical fitness trajectory, identical generation of improvement (gen 2),
identical final value. Diversity (§8.2) is within noise between the two
arms as well. **For 156 extra tokens (0.5% more than `nocomm`'s total),
communication produced no measurable difference in this run.** That is
the honest headline this run supports: on this substrate, at this scale,
communication did not pay for itself. (`off` stayed flat at 1.0 the whole
run -- gained=0, so cost-per-fitness-point is undefined there, not forced
via a proxy. Any comparison of `off` against the two agent arms remains
confounded by call budget as noted above, so this section does not claim
agents beat the baseline -- only that, once agents are running, adding
communication on top bought nothing measurable for its price.)

### 8.2 Diversity comes from the role, not the communication

This is what the third arm was built to isolate, and it answers cleanly:

```
off:     [10.3, 6.3, 2.7, 3.1, 2.6, 3.3]    -- collapses by gen 2, stays low
nocomm:  [10.3, 7.86, 7.43, 7.1,  8.57, 5.38]
comm:    [10.3, 7.86, 7.43, 7.14, 8.14, 5.81]
```

`nocomm` and `comm` track each other closely throughout (differences of
0.04-0.4 from generation 3 on -- noise-level, plausibly just downstream
RNG drift once communication starts touching prompts, not a real
divergence), while both sit roughly 2-3x above `off` at every generation
past the first. Communication was never the mechanism holding diversity
up in the first smoke run's off/on comparison -- the explore/refine roles
were, with or without agents ever exchanging a message. §8.3 narrows this
further: given refine's behavior below, "the roles" here reduces to "the
explore role, specifically."

### 8.3 Refine is a self-sustaining no-op

Every refine call, in both arms, produced the exact same output, verbatim,
for all 6 generations:

```
gen=0: Your current fold: L S U L S S D U D S R S  ->  FOLD: L S U L S S D U D S R S
gen=1: Your current fold: L S U L S S D U D S R S  ->  FOLD: L S U L S S D U D S R S
...    (identical through gen=5, in both nocomm and comm)
```

Every one of these is `valid=True` on the first attempt -- not a parse
failure hiding behind a fallback, a "successful" response by the format
contract that is a complete content no-op. **The mechanism is a fixed
point, stated explicitly**: refine's output becomes its own next
generation's input (population[pop_size + agent_id] is read back
unchanged as "your current fold"), so the moment it echoes its input once,
every subsequent call is handed that same echo and echoes it again --
permanently, with no way out under this prompt. It happened as early as
generation 0 here (the random starting fold happened to be echoed
immediately), but the mechanism guarantees it happens *eventually*
regardless of the starting point, the first time the model takes the
cheapest valid path. Unlike crossover, which rejects and retries a
near-echo via `_crossover_sufficiently_mixed` (`CROSSOVER_MIN_DIFF`), the
refine prompt has no min-diff guard -- nothing in this design would ever
catch or retry against this.

Practically: **refine contributed zero distinct genomes across the entire
run, in either arm. `A=2` therefore ran as `A=1` the whole time** -- every
effect reported in §8.1 and §8.2 is attributable to the explore agent
alone (confirmed independently: explore produced 6/6 unique responses per
arm, touching most-to-all of the 12 positions repeatedly, no collapse).
Half the agent token budget bought nothing at all, on top of buying no
measurable benefit from communication (§8.1).

This is not a new failure mode -- it's the same one. §4.1 documented
`mutate/full`'s silent no-ops (the cheapest valid answer under a
free-form, restate-everything format is to copy the input), and §4.3's
affordance fix (change what the model is asked to produce, not how it's
worded) was built specifically to route around it. The agent roles use
the same free-form, restate-the-whole-fold affordance as `mutate/full`
(by design, for a first pass -- no position/segment-equivalent affordance
was built for agents), and refine reproduces the identical failure,
except worse: `mutate/full`'s no-ops varied call to call (copy *this*
input, whatever it currently was) with no persistence, while refine's
identical prompt/response loop turns a single no-op into a permanent one.
**This strengthens the affordance finding rather than repeating it**: the
same format choice that was already known to be the weak affordance for
mutation produces a more severe, self-locking version of the same failure
when the output of one call becomes the input to the next -- exactly the
persistent-identity structure this feature introduces. A position- or
diff-style affordance for refine (analogous to §4.3's fix) is the
concrete next step this result points to, not a re-run at the same
settings.
