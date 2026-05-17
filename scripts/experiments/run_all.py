"""Run variants A, B, C through scripts/run_variant.py and emit a comparison CSV.

CSV columns:
    variant, overall_brier, n, wall_seconds, <category>:brier, <category>:n

Baseline row is read from backtest_results.json (the existing main.py run).

Usage:
    python scripts/experiments/run_all.py [workers]
"""

from __future__ import annotations

import csv
import io
import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO))

from scripts.run_variant import run  # noqa: E402

VARIANTS = [
    ("variant_a", "scripts.experiments.variant_a", "results_a.json"),
    ("variant_b", "scripts.experiments.variant_b", "results_b.json"),
    ("variant_c", "scripts.experiments.variant_c", "results_c.json"),
]


def baseline_summary() -> dict | None:
    p = _REPO / "backtest_results.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text(encoding="utf-8"))
    by_cat_n: dict[str, int] = {}
    for r in d.get("results", []):
        by_cat_n[r["category"]] = by_cat_n.get(r["category"], 0) + 1
    return {
        "module": "main (baseline)",
        "overall_brier": d.get("overall_brier"),
        "n": d.get("n"),
        "wall_seconds": None,
        "by_category": d.get("by_category", {}),
        "by_category_n": by_cat_n,
    }


def main(workers: int = 5) -> None:
    events_path = str(_REPO / "events_resolved.json")
    actuals_path = str(_REPO / "actuals.json")
    out_dir = _REPO / "scripts" / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries: list[dict] = []

    baseline = baseline_summary()
    if baseline:
        summaries.append({"variant": "baseline_main", **baseline})

    for name, module_path, out_file in VARIANTS:
        out_path = str(out_dir / out_file)
        print(f"\n>>> Running {name} ({module_path}) ...")
        try:
            s = run(module_path, events_path, actuals_path, workers, out_path)
            s["variant"] = name
            summaries.append(s)
        except Exception as exc:
            print(f"!!! {name} crashed: {type(exc).__name__}: {exc}")
            summaries.append({
                "variant": name, "module": module_path,
                "overall_brier": None, "n": 0, "wall_seconds": None,
                "by_category": {}, "by_category_n": {},
                "error": f"{type(exc).__name__}: {exc}",
            })

    # All categories observed across runs
    cats: set[str] = set()
    for s in summaries:
        cats.update(s.get("by_category", {}).keys())
    cats_sorted = sorted(cats)

    csv_path = out_dir / "results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        header = ["variant", "module", "overall_brier", "n", "wall_seconds"]
        for c in cats_sorted:
            header += [f"{c}:brier", f"{c}:n"]
        header.append("error")
        w.writerow(header)
        for s in summaries:
            row = [
                s.get("variant"),
                s.get("module"),
                _fmt(s.get("overall_brier")),
                s.get("n"),
                _fmt(s.get("wall_seconds")),
            ]
            for c in cats_sorted:
                row.append(_fmt(s.get("by_category", {}).get(c)))
                row.append(s.get("by_category_n", {}).get(c, 0))
            row.append(s.get("error", ""))
            w.writerow(row)

    print(f"\n=== Summary written to {csv_path} ===")
    print(f"{'variant':<18}{'overall_brier':>15}{'n':>5}")
    for s in summaries:
        b = s.get("overall_brier")
        b_str = f"{b:.4f}" if isinstance(b, (int, float)) else str(b)
        print(f"{s.get('variant',''):<18}{b_str:>15}{s.get('n',0):>5}")


def _fmt(v) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.6f}"
    return str(v)


if __name__ == "__main__":
    w = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    main(w)
