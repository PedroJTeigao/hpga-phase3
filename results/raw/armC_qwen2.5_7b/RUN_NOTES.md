# Run notes: arm C on qwen2.5:7b (written by experiments/run_arm_c_model.py)

Exit reason: 'all planned runs attempted'. Done: [(0, 1), (1, 1), (2, 1), (3, 1), (4, 1)]. Skipped (already complete): [].

## Ollama model loads (load_duration from each Ollama response; >= 1 s counts as a load)

Wall time and llm_s in these runs include these loads. Loads during these runs took 3.98-50.19 s (table below; the 50.19 s load was the first of seed 0, every later load was 4.0-9.2 s), against 14.7 s on 2026-09-21; the pre-launch test loads on 2026-09-30 took 127-181 s. Wall time here is not comparable with any earlier run; the budget (distinct folds) and every fitness and behaviour measure are unaffected.

| seed | loads | min s | max s | total s |
|---|---|---|---|---|
| 0 | 19 | 4.11 | 50.19 | 151.4 |
| 1 | 19 | 4.04 | 7.53 | 98.2 |
| 2 | 19 | 4.09 | 9.22 | 101.3 |
| 3 | 19 | 4.0 | 5.89 | 86.5 |
| 4 | 19 | 3.98 | 7.18 | 87.0 |

Observed range over all loads: 3.98 - 50.19 s (95 loads). Every event: ollama_loads.jsonl.
