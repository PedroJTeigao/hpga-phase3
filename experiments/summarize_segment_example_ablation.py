"""Summarise the --segment-example ablation of probe_sequence_operator_compliance.py
from its raw output JSONs, and record it as results/raw/segment_example_ablation_summary.json.

The result this exists to state: how much of the boundary space the segment
crossover operator ever reaches. Any integer cut 1..99 is a valid two-segment
declaration for every pair of parent lengths in [MIN_LENGTH, MAX_LENGTH]
(checked exhaustively: 0 rejections in 51*51*99 combinations), so the space is
N_POSSIBLE_CUTS = 99 values; "distinct values chosen" is measured against that.
Settings are reported one by one; rates are never pooled. The only cross-setting
figure is the SET of values ever chosen, which is a union, not a rate.

Read-only on the raw files; touches nothing else.
"""

import json
from pathlib import Path

RAW = Path(__file__).resolve().parent.parent / "results" / "raw"
SETTINGS = ("current", "shifted", "absent", "absent_prose25")
N_POSSIBLE_CUTS = 99


def load(setting: str) -> dict:
    path = RAW / f"sequence_operator_compliance_gemma4_12b_len63_segex_{setting}.json"
    d = json.load(open(path, encoding="utf-8"))
    assert d["complete"] and d["segment_example"] == setting, path
    (c,) = [c for c in d["conditions"] if c["op"] == "crossover" and c["style"] == "segment"]
    return {"header": d, "cond": c, "file": path.name}


def main() -> None:
    per_setting, all_values = {}, {}
    for s in SETTINGS:
        r = load(s)
        c, b, h = r["cond"], r["cond"]["boundary_summary"], r["header"]
        counts = {int(k): v for k, v in b["cut_percent_counts"].items()}
        per_setting[s] = {
            "source_file": r["file"], "model": h["model"], "temperature": h["temperature"], "seed": h["seed"],
            "genome_len": h["genome_len"], "length_bounds": h["length_bounds"],
            "n_calls": c["n_calls"], "fallback_rate": c["fallback_rate"], "requests_per_call": c["requests_per_call"],
            "n_cuts": b["n_cuts"], "cut_counts": dict(sorted(counts.items())),
            "n_distinct_values": len(counts), "distinct_values": sorted(counts),  # from the counts: the first three JSONs predate the field
            "share_of_space_reached": len(counts) / N_POSSIBLE_CUTS,
            "modal_values": b["modal_values"], "modal_share": b["modal_share_of_cuts"],
            "segment_example_check": c["segment_example_check"],
        }
        all_values[s] = set(counts)

    first3 = set().union(*(all_values[s] for s in SETTINGS[:3]))
    every = set().union(*all_values.values())
    summary = {
        "n_possible_cut_values": N_POSSIBLE_CUTS,
        "possible_values_note": "integer cuts 1..99; every one is valid for every parent-length pair in the length bounds",
        "per_setting": per_setting,
        "values_ever_chosen": {
            "first_three_settings_300_calls": sorted(first3),
            "all_four_settings_400_calls": sorted(every),
        },
    }
    out = RAW / "segment_example_ablation_summary.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"boundary space: {N_POSSIBLE_CUTS} possible cut values (1..99)\n")
    print(f"{'setting':<15}{'calls':>6}{'fallback':>10}{'req/call':>10}{'distinct':>10}  {'values (count)':<28}{'modal (share)'}")
    for s, p in per_setting.items():
        vals = ", ".join(f"{v}x{n}" for v, n in p["cut_counts"].items())
        modal = "/".join(map(str, p["modal_values"]))
        print(f"{s:<15}{p['n_calls']:>6}{p['fallback_rate']:>10.2f}{p['requests_per_call']:>10.2f}"
              f"{p['n_distinct_values']:>7}/{N_POSSIBLE_CUTS}  {vals:<28}{modal} ({p['modal_share']:.0%})")
    print(f"\nvalues ever chosen, settings 1-3 (300 calls): {sorted(first3)}")
    print(f"values ever chosen, all four settings (400 calls): {sorted(every)}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
