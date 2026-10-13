# Spark probe handoff, 2026-10-06

Both probes completed with worker return code 0 using
`nvidia/Qwen3.6-35B-A3B-NVFP4` at the Spark OpenAI-compatible endpoint.
This model differs from the 26B model requested in the email; the result
should be identified by its exact model name rather than treated as a
controlled model-scale comparison.

Configuration recorded in both metadata files: temperature 0.7, two extra
attempts allowed, 600-second request timeout, non-thinking Qwen template,
and PYTHONHASHSEED=0. The configured context_tokens=4096 is a legacy client
setting; it is not sent as a context limit to vLLM. The server manages its
context capacity. No folding or GA ran. There is no recorded model weight
digest from the remote API.

## Completion checks

| Directory | Operator calls | Logged HTTP attempts | Logged deterministic fallbacks | Transport errors | Elapsed time |
|---|---:|---:|---:|---:|---|
| oldfield_gate_spark_qwen | 1000 | 1593 | 205 | 0 | 16 min 49 s |
| numbered_probe_spark_qwen | 900 | 911 | 2 | 0 | 9 min 31 s |

The attempts above exclude each launcher's warmup request. All JSONL files
parsed successfully. Per-cell first-attempt counts, unique input counts,
attempt counts, and fallback-event counts agree with the worker records.

| Baseline cell | Operator calls | HTTP attempts | Fallbacks |
|---|---:|---:|---:|
| C1_old | 100 | 102 | 0 |
| C1_new | 100 | 101 | 0 |
| C2_old | 200 | 202 | 0 |
| C2_new | 200 | 495 | 104 |
| C3_old | 200 | 206 | 0 |
| C3_new | 200 | 487 | 101 |

| Numbered cell | Operator calls | HTTP attempts | Fallbacks |
|---|---:|---:|---:|
| C1_drift | 100 | 101 | 0 |
| C2_num | 200 | 200 | 0 |
| C2_numold | 200 | 205 | 1 |
| C3_num | 200 | 200 | 0 |
| C3_numold | 200 | 205 | 1 |

These are execution checks using existing stats and log events, not a
scientific rescore or a classification of model errors. The baseline OLD
field cells account for all 205 baseline fallbacks.

## Input checks and scoring caveat

All 100 first-attempt C1 prompts match between C1_old and C1_drift.
Their complete attempt streams have different lengths (102 versus 101)
because retry counts differ. The strict preregistered all-attempt drift
check therefore cannot report identical streams; Pedro should decide
whether and how the original reuse rule applies.

The 200 C2 first-attempt input sequences match between spaced and numbered
displays. The 200 C3 first-attempt input sequences also match. The generated
baseline replay list matches the committed replay list used by the numbered
probe on genome, k, source, and log fields.

No historical summarizer was run and no raw output was modified. Both
original output directories and this note are included in the handoff ZIP.
The files have been prepared locally; nothing has been sent to Pedro.
