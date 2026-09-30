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


_orig_base_record, _orig_run_ga = base.base_record, base.run_ga


def base_record(arm, seed, args):
    rec = _orig_base_record(arm, seed, args)
    rec["provenance"] = {"wrapper": "experiments/run_arm_c_model.py", "at_start": snapshot()}
    return rec


def run_ga(arm, seed, args, llm):
    log_path = Path(args.out_dir) / f"llm_operator_calls_armC_{ops.LLM_MODEL.replace(':', '_')}_seed{seed}_{int(time.time())}.jsonl"
    os.environ["HPGA_LLM_LOG_PATH"] = str(log_path)
    try:
        rec = _orig_run_ga(arm, seed, args, llm)
    finally:
        os.environ.pop("HPGA_LLM_LOG_PATH", None)
    end = snapshot()
    prov = rec["provenance"]
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
    Path(known.out_dir).mkdir(parents=True, exist_ok=True)
    base.base_record, base.run_ga = base_record, run_ga
    base.main()


if __name__ == "__main__":
    main()
