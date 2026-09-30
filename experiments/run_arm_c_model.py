"""Arm C with a different LLM, plus model provenance: run_sequence_ga_comparison.py's arm C, unchanged, with the model
build recorded in every result file.

Why a wrapper. Arms C (2026-09-18/19) and F (2026-09-28) ran under the same tag, gemma4:12b, and the run files record only
the tag, so "the same model build served both" could only be inferred afterwards from file times. This wrapper records the
build. It changes nothing in the driver: it wraps two module-level functions of run_sequence_ga_comparison.py from outside
(the same monkeypatch pattern the drivers already use on hpga.operators). The driver looks both up by name at call time.

  base_record  gains rec["provenance"]["at_start"]: the Ollama server version, the tag's manifest digest, modified_at and
               size (/api/tags), its details (/api/show: family, parameter size, quantization), the host, and the LLM
               settings the operators read from the environment.
  run_ga       gains rec["provenance"]["at_end"], the same snapshot taken after the run, and "digest_unchanged". A change
               in the middle of a run is recorded and logged, not raised, so an hour's run is not discarded; it must be
               read before the run is used. Also: each seed's LLM call log goes into --out-dir, named with the model
               (HPGA_LLM_LOG_PATH), instead of results/raw/llm_operator_calls_seqcmp_C_seed<n>_*.jsonl, which does not
               carry the model name and would sit next to the gemma logs.

Ollama load times: the Ollama client's generate() is wrapped to record each response's load_duration (>= 1 s counts as
a load), because arm C's accounting hides reloads inside LLM time. Every load goes to --out-dir/ollama_loads.jsonl and
to rec["provenance"]["ollama_loads"] for its seed. On exit, normal or not, --out-dir/RUN_NOTES.md is written with the
per-seed load range and every seed that failed twice and was skipped.

Refuses anything but arm C, and refuses --out-dir = results/raw: there the driver would skip every seed, because the gemma
files sequence_ga_cmp_C_seed<n>.json already exist and are complete, and it would overwrite the committed
sequence_ga_comparison_status.json.

  PYTHONHASHSEED=0 HPGA_LLM_MODEL=qwen2.5:7b HPGA_LLM_TIMEOUT_S=600 \\
    python experiments/run_arm_c_model.py run --arms C --seeds 0 1 2 3 4 --out-dir results/raw/armC_qwen2.5_7b

HPGA_LLM_MODEL must be set explicitly; relying on the driver's default (gemma4:12b) is refused.
"""

import argparse
import json
import logging
import os
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

if not os.environ.get("HPGA_LLM_MODEL"):
    raise SystemExit("set HPGA_LLM_MODEL explicitly (hpga/operators.py reads it once, at import)")

import run_sequence_ga_comparison as base  # noqa: E402  (imports hpga.operators, which reads HPGA_LLM_* now)

from hpga import operators as ops  # noqa: E402

log = logging.getLogger("armc_model")
ENV_KEYS = ("HPGA_LLM_MODEL", "HPGA_OLLAMA_HOST", "HPGA_LLM_TEMPERATURE", "HPGA_LLM_NUM_CTX", "HPGA_LLM_KEEP_ALIVE",
            "HPGA_LLM_TIMEOUT_S", "HPGA_LLM_MAX_RETRIES", "PYTHONHASHSEED")


def _http_json(path: str, payload: dict | None = None):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(ops.LLM_HOST.rstrip("/") + path, data=data,
                                 headers={"Content-Type": "application/json"} if data else {})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def snapshot() -> dict:
    """What served this run. Every field is read, never assumed; a failed read is recorded as an error string."""
    snap = {"time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "host": ops.LLM_HOST, "tag": ops.LLM_MODEL,
            "operators_settings": {"temperature": ops.LLM_TEMPERATURE, "num_ctx": ops.LLM_NUM_CTX,
                                   "keep_alive": ops.LLM_KEEP_ALIVE, "timeout_s": ops.LLM_REQUEST_TIMEOUT_S,
                                   "max_retries": ops.LLM_MAX_RETRIES},
            "env": {k: os.environ.get(k) for k in ENV_KEYS}}
    try:
        snap["ollama_version"] = _http_json("/api/version").get("version")
    except Exception as exc:  # noqa: BLE001
        snap["ollama_version"] = f"<error: {exc!r}>"
    try:
        tags = _http_json("/api/tags").get("models", [])
        entry = next((m for m in tags if m.get("name") == ops.LLM_MODEL or m.get("model") == ops.LLM_MODEL), None)
        snap["model_digest"] = entry.get("digest") if entry else "<tag not found in /api/tags>"
        snap["model_modified_at"] = entry.get("modified_at") if entry else None
        snap["model_size_bytes"] = entry.get("size") if entry else None
    except Exception as exc:  # noqa: BLE001
        snap["model_digest"] = f"<error: {exc!r}>"
    try:
        snap["model_details"] = _http_json("/api/show", {"model": ops.LLM_MODEL}).get("details")
    except Exception as exc:  # noqa: BLE001
        snap["model_details"] = f"<error: {exc!r}>"
    return snap


# --- Ollama load times --------------------------------------------------------------------------------------------
# Arm C's accounting puts each Ollama reload inside the first LLM call of a breeding step, and operators._call_ollama
# keeps only that call's total latency, so the reload itself is not recorded anywhere (SEQUENCE_GA_REPORT.md sec 9.7
# had to estimate it). Ollama reports it per response as load_duration. The client's generate() is wrapped to record
# that field and return the response untouched: same calls, same order, same return values.

LOAD_EVENT_MIN_S = 1.0  # a request against an already-resident model reports a load_duration of milliseconds
_load_events: list[dict] = []
_current_seed: int | None = None
_loads_path: Path | None = None
_orig_get_client = ops._get_client


def _get_client():
    client = _orig_get_client()
    if not getattr(client, "_armc_load_recorder", False):
        orig_generate = client.generate

        def generate(*a, **k):
            resp = orig_generate(*a, **k)
            unload = k.get("keep_alive") == 0
            load_s = (getattr(resp, "load_duration", None) or 0) / 1e9
            if not unload and load_s >= LOAD_EVENT_MIN_S:
                ev = {"time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "seed": _current_seed, "load_s": round(load_s, 2),
                      "total_s": round((getattr(resp, "total_duration", None) or 0) / 1e9, 2)}
                _load_events.append(ev)
                if _loads_path is not None:
                    with open(_loads_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(ev) + "\n")
            return resp

        client.generate = generate
        client._armc_load_recorder = True
    return client


def _load_summary(events: list[dict]) -> dict:
    s = [e["load_s"] for e in events]
    return {"n_loads": len(s), "min_s": min(s) if s else None, "max_s": max(s) if s else None,
            "total_s": round(sum(s), 1), "threshold_s": LOAD_EVENT_MIN_S, "events": events}


def write_run_notes(out_dir: Path) -> None:
    """RUN_NOTES.md in --out-dir: Ollama load times per seed and any failed or skipped seed, so the wall-time caveat
    travels with the files."""
    status_p = out_dir / "sequence_ga_comparison_status.json"
    status = json.load(open(status_p)) if status_p.exists() else {}
    lines = [f"# Run notes: arm C on {ops.LLM_MODEL} (written by experiments/run_arm_c_model.py)", "",
             f"Exit reason: {status.get('exit_reason')!r}. Done: {[(d['seed'], d['attempt']) for d in status.get('done', [])]}. "
             f"Skipped (already complete): {[d['seed'] for d in status.get('skipped', [])]}.", ""]
    if status.get("failed"):
        lines += ["**FAILED SEEDS (failed on both attempts; the driver skipped them and moved on):**", ""]
        lines += [f"- seed {f['seed']}: {f['error']}" for f in status["failed"]] + [""]
    lines += ["## Ollama model loads (load_duration from each Ollama response; >= 1 s counts as a load)", "",
              "Wall time and llm_s in these runs include these loads. Loads on 2026-09-30 took 127-181 s against "
              "14.7 s on 2026-09-21, so wall time here is not comparable with any earlier run; the budget (distinct "
              "folds) and every fitness and behaviour measure are unaffected.", "",
              "| seed | loads | min s | max s | total s |", "|---|---|---|---|---|"]
    by_seed: dict = {}
    for e in _load_events:
        by_seed.setdefault(e["seed"], []).append(e)
    for seed, evs in sorted(by_seed.items(), key=lambda kv: (kv[0] is None, kv[0])):
        s = _load_summary(evs)
        lines.append(f"| {seed if seed is not None else 'outside a run'} | {s['n_loads']} | {s['min_s']} | {s['max_s']} | {s['total_s']} |")
    allv = [e["load_s"] for e in _load_events]
    lines += ["", f"Observed range over all loads: {min(allv) if allv else None} - {max(allv) if allv else None} s "
                  f"({len(allv)} loads). Every event: ollama_loads.jsonl."]
    (out_dir / "RUN_NOTES.md").write_text("\n".join(lines) + "\n")


_orig_base_record, _orig_run_ga = base.base_record, base.run_ga


def base_record(arm, seed, args):
    rec = _orig_base_record(arm, seed, args)
    rec["provenance"] = {"wrapper": "experiments/run_arm_c_model.py", "at_start": snapshot()}
    return rec


def run_ga(arm, seed, args, llm):
    global _current_seed
    log_path = Path(args.out_dir) / f"llm_operator_calls_armC_{ops.LLM_MODEL.replace(':', '_')}_seed{seed}_{int(time.time())}.jsonl"
    os.environ["HPGA_LLM_LOG_PATH"] = str(log_path)
    _current_seed, n_before = seed, len(_load_events)
    try:
        rec = _orig_run_ga(arm, seed, args, llm)
    finally:
        os.environ.pop("HPGA_LLM_LOG_PATH", None)
        _current_seed = None
    end = snapshot()
    prov = rec["provenance"]
    prov["ollama_loads"] = _load_summary(_load_events[n_before:])
    prov["at_end"] = end
    prov["digest_unchanged"] = prov["at_start"].get("model_digest") == end.get("model_digest")
    prov["ollama_version_unchanged"] = prov["at_start"].get("ollama_version") == end.get("ollama_version")
    if not (prov["digest_unchanged"] and prov["ollama_version_unchanged"]):
        log.error("seed %d: model digest or Ollama version CHANGED during the run: %s -> %s", seed,
                  (prov["at_start"].get("model_digest"), prov["at_start"].get("ollama_version")),
                  (end.get("model_digest"), end.get("ollama_version")))
    return rec


def main() -> None:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("cmd")
    ap.add_argument("--arms", nargs="+")
    ap.add_argument("--out-dir")
    known, _ = ap.parse_known_args()
    if known.cmd != "run" or known.arms != ["C"]:
        raise SystemExit("this wrapper runs arm C only: use `run --arms C --seeds ... --out-dir DIR`")
    if not known.out_dir or Path(known.out_dir).resolve() == base.RAW.resolve():
        raise SystemExit("--out-dir must be given and must not be results/raw (the gemma arm C files live there)")
    global _loads_path
    out_dir = Path(known.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _loads_path = out_dir / "ollama_loads.jsonl"
    base.base_record, base.run_ga = base_record, run_ga
    ops._get_client = _get_client
    try:
        base.main()
    finally:
        write_run_notes(out_dir)


if __name__ == "__main__":
    main()
