"""Arm C with another model (experiments/run_arm_c_model.py) against arm C on gemma4:12b: provenance gate, then best
TM-score at the common distinct-fold count per seed.

THE GATE RUNS FIRST, before any number is computed, on every complete result file in --model-dir. It refuses (exit 1,
listing every failing file and condition) unless all four hold:

  1. the file has a "provenance" block (it was produced through run_arm_c_model.py);
  2. provenance.digest_unchanged and provenance.ollama_version_unchanged are both true (the build did not change during
     the run);
  3. the digest and the Ollama version recorded at start AND at end are real values: the digest is 64 hex characters
     (an optional "sha256:" prefix is allowed), and the version is not an error string. Without this, two identical
     error placeholders (e.g. "<tag not found in /api/tags>") would compare equal and pass condition 2 while checking
     nothing;
  4. every file has the same start digest and the same start Ollama version: one build served the whole arm, not just
     each run.

--allow-provenance-mismatch lets a failing set through. It prints a banner to stderr and writes the failures, verbatim,
at the top of the generated tables, so a bypassed run cannot pass silently. The gemma files predate provenance
recording and are not gated; the tables say so.

Comparison: per seed, n_cut = the smaller distinct-fold count of the two runs; each arm's best-so-far at n_cut; the
project's fixed rule (X beats Y only if higher in every seed). This file evaluates nothing from
results/PREREGISTERED_QWEN.md; that is a separate step.

  python experiments/summarize_arm_c_model.py --model-dir results/raw/armC_qwen2.5_7b \\
      --out-md results/raw/armC_qwen2.5_7b/armC_model_tables.md
"""

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "results" / "raw"
_DIGEST = re.compile(r"^(sha256:)?[0-9a-f]{64}$")


def load_complete(directory: Path) -> dict[int, tuple[Path, dict]]:
    runs = {}
    for p in sorted(directory.glob("sequence_ga_cmp_C_seed*.json")):
        d = json.load(open(p))
        if d.get("complete"):
            runs[int(d["seed"])] = (p, d)
    return runs


def _real_digest(v) -> bool:
    return isinstance(v, str) and bool(_DIGEST.match(v))


def _real_version(v) -> bool:
    return isinstance(v, str) and bool(v) and not v.startswith("<")


def provenance_failures(runs: dict[int, tuple[Path, dict]]) -> list[str]:
    """Every failure of conditions 1-4, one line each. Empty list = the set passes."""
    fails, starts = [], {}
    for seed, (path, d) in sorted(runs.items()):
        name = path.name
        prov = d.get("provenance")
        if not isinstance(prov, dict):  # 1
            fails.append(f"{name}: no provenance block (not produced by run_arm_c_model.py)")
            continue
        for flag in ("digest_unchanged", "ollama_version_unchanged"):  # 2
            if prov.get(flag) is not True:
                fails.append(f"{name}: provenance.{flag} = {prov.get(flag)!r}")
        for when in ("at_start", "at_end"):  # 3
            snap = prov.get(when) or {}
            if not _real_digest(snap.get("model_digest")):
                fails.append(f"{name}: {when}.model_digest is not a real digest: {snap.get('model_digest')!r}")
            if not _real_version(snap.get("ollama_version")):
                fails.append(f"{name}: {when}.ollama_version is not a real version: {snap.get('ollama_version')!r}")
        start = prov.get("at_start") or {}
        starts[name] = (start.get("model_digest"), start.get("ollama_version"))
    if len(set(starts.values())) > 1:  # 4
        listing = "; ".join(f"{n}: digest {dg!r}, ollama {ver!r}" for n, (dg, ver) in sorted(starts.items()))
        fails.append(f"not one build across seeds: {listing}")
    return fails


def best_at(d: dict, n: int) -> float:
    return d["best_so_far_by_distinct_evaluation"][n - 1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True, help="the --out-dir given to run_arm_c_model.py")
    ap.add_argument("--gemma-dir", default=str(RAW), help="where arm C on gemma4:12b lives (default results/raw)")
    ap.add_argument("--out-md", required=True)
    ap.add_argument("--allow-provenance-mismatch", action="store_true")
    args = ap.parse_args()

    model_runs = load_complete(Path(args.model_dir))
    gemma_runs = load_complete(Path(args.gemma_dir))
    if not model_runs:
        sys.exit(f"no complete sequence_ga_cmp_C_seed*.json in {args.model_dir}")

    fails = provenance_failures(model_runs)
    banner = []
    if fails:
        msg = "PROVENANCE CHECK FAILED:\n" + "\n".join(f"  - {f}" for f in fails)
        if not args.allow_provenance_mismatch:
            sys.exit(msg + "\nRefusing to summarise. --allow-provenance-mismatch overrides, and records that it did.")
        bar = "!" * 78
        print(f"{bar}\n{msg}\nCONTINUING ONLY BECAUSE --allow-provenance-mismatch WAS GIVEN.\n{bar}", file=sys.stderr)
        banner = ["> **PROVENANCE CHECK FAILED; these tables were generated with `--allow-provenance-mismatch`.**",
                  "> Do not cite any number below without reading these failures:", ">"] + [f"> - {f}" for f in fails] + [""]

    any_prov = next(iter(model_runs.values()))[1]["provenance"] if not fails else None
    tag = next(iter(model_runs.values()))[1]["config"]["llm_model"]
    lines = [f"# Arm C: {tag} vs gemma4:12b (generated by experiments/summarize_arm_c_model.py)", ""] + banner
    lines += ["## Provenance", ""]
    if any_prov:
        s = any_prov["at_start"]
        lines += [f"- {tag}: digest `{s['model_digest']}`, Ollama {s['ollama_version']}, details {s.get('model_details')}; "
                  f"identical at start and end of all {len(model_runs)} runs (all four checks passed).", ""]
    lines += ["- gemma4:12b arm C files predate provenance recording: only the tag is recorded, not gated.", ""]

    seeds = sorted(set(model_runs) & set(gemma_runs))
    missing = sorted(set(model_runs) ^ set(gemma_runs))
    lines += ["## Best TM-score at the common distinct-fold count", "",
              f"| seed | {tag} distinct | gemma distinct | n_cut | {tag} best@n_cut | gemma best@n_cut | difference | higher |",
              "|---|---|---|---|---|---|---|---|"]
    diffs = []
    for s in seeds:
        m, g = model_runs[s][1], gemma_runs[s][1]
        nm, ng = m["summary"]["n_distinct_evaluations"], g["summary"]["n_distinct_evaluations"]
        n = min(nm, ng)
        bm, bg = best_at(m, n), best_at(g, n)
        diffs.append(bm - bg)
        higher = tag if bm > bg else ("gemma4:12b" if bg > bm else "tie")
        lines.append(f"| {s} | {nm} | {ng} | {n} | {bm:.4f} | {bg:.4f} | {bm - bg:+.4f} | {higher} |")
    if missing:
        lines += ["", f"Seeds present in only one arm (not compared): {missing}"]
    if diffs:
        up, down = sum(x > 0 for x in diffs), sum(x < 0 for x in diffs)
        k = len(diffs)
        verdict = (f"{tag} BEATS gemma4:12b under the fixed rule" if up == k else
                   f"gemma4:12b BEATS {tag} under the fixed rule" if down == k else "NO DEMONSTRATED DIFFERENCE")
        lines += ["", f"**{verdict}**: {tag} higher in {up}, lower in {down}, tied in {k - up - down} of {k} seeds; "
                      f"mean difference {sum(diffs) / k:+.4f}. Smallest two-sided sign-test p possible with {k} seeds: "
                      f"{2 * 0.5 ** k:.4g}."]
    out = Path(args.out_md)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
