"""Run the parallel Brier backtest against an arbitrary variant module.

The variant module must expose `predict(event: dict) -> dict` with the same
contract as `main.predict`. Existing `scripts/backtest.py` is left untouched.

Usage:
    python scripts/run_variant.py <module> [events.json] [actuals.json] [workers] [out.json]

Example:
    python scripts/run_variant.py scripts.experiments.variant_a \
        events_resolved.json actuals.json 5 scripts/experiments/results_a.json
"""

from __future__ import annotations

import importlib
import io
import json
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Force UTF-8 stdout on Windows cp949 consoles
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# Make repo root importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def brier(probs_by_outcome: dict, outcomes: list, actual: str) -> float:
    s = 0.0
    for outcome in outcomes:
        p = probs_by_outcome.get(outcome, 0.0)
        target = 1.0 if outcome == actual else 0.0
        s += (p - target) ** 2
    return s


def score_event(predict_fn, event: dict, actual: str) -> dict:
    t0 = time.time()
    pred = predict_fn(event)
    elapsed = time.time() - t0
    probs = {p["market"]: float(p["probability"]) for p in pred["probabilities"]}
    b = brier(probs, event["outcomes"], actual)
    return {
        "ticker": event["market_ticker"],
        "title": event["title"][:80],
        "category": event["category"],
        "n_outcomes": len(event["outcomes"]),
        "actual": actual,
        "brier": b,
        "elapsed_s": elapsed,
        "predicted": probs,
        "rationale": pred.get("rationale", ""),
    }


def run(module_path: str,
        events_path: str = "events_resolved.json",
        actuals_path: str = "actuals.json",
        workers: int = 5,
        out_path: str | None = None) -> dict:
    mod = importlib.import_module(module_path)
    predict_fn = mod.predict

    events = json.loads(Path(events_path).read_text(encoding="utf-8"))
    actuals = json.loads(Path(actuals_path).read_text(encoding="utf-8"))

    pairs = [(e, actuals[e["market_ticker"]]) for e in events
             if e["market_ticker"] in actuals]
    print(f"[{module_path}] Scoring {len(pairs)} resolved events with {workers} parallel workers")

    results: list[dict] = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(score_event, predict_fn, e, a): e["market_ticker"]
                   for e, a in pairs}
        for i, fut in enumerate(as_completed(futures), 1):
            try:
                r = fut.result()
            except Exception as exc:
                print(f"  [{i}/{len(pairs)}] {futures[fut]} ERROR: {exc}")
                continue
            results.append(r)
            tag = "+" if r["brier"] < 0.25 else "-"
            print(f"  [{i}/{len(pairs)}] {tag} {r['ticker']}  Brier={r['brier']:.4f}  ({r['elapsed_s']:.1f}s)")

    total = time.time() - t0
    if not results:
        print(f"\n[{module_path}] No events scored.")
        return {"overall_brier": float("nan"), "n": 0, "by_category": {}, "results": []}

    overall = sum(r["brier"] for r in results) / len(results)
    by_cat: dict[str, list[float]] = defaultdict(list)
    for r in results:
        by_cat[r["category"]].append(r["brier"])

    print(f"\n{'='*60}")
    print(f"[{module_path}] Brier: {overall:.4f}  (n={len(results)}, wall={total:.1f}s)")
    print(f"{'='*60}")
    for cat, briers in sorted(by_cat.items()):
        print(f"  {cat:<20} n={len(briers):2}  avg={sum(briers)/len(briers):.4f}")

    summary = {
        "module": module_path,
        "overall_brier": overall,
        "n": len(results),
        "wall_seconds": total,
        "by_category": {k: sum(v) / len(v) for k, v in by_cat.items()},
        "by_category_n": {k: len(v) for k, v in by_cat.items()},
        "results": results,
    }
    if out_path:
        Path(out_path).write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"\n[{module_path}] Full results -> {out_path}")
    return summary


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        print("Usage: run_variant.py <module> [events.json] [actuals.json] [workers] [out.json]")
        sys.exit(2)
    module_path = args[0]
    events = args[1] if len(args) > 1 else "events_resolved.json"
    actuals = args[2] if len(args) > 2 else "actuals.json"
    workers = int(args[3]) if len(args) > 3 else 5
    out = args[4] if len(args) > 4 else None
    run(module_path, events, actuals, workers, out)
