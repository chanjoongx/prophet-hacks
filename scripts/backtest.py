"""Parallel local Brier backtest against resolved sample data.

Bypasses the CLI (which skips past-close events) by importing predict()
directly. Calls the LLM in parallel and reports overall + per-category Brier.

Usage:
    python scripts/backtest.py [events.json] [actuals.json] [workers]
"""

from __future__ import annotations

import io
import json
import sys
import time

# Force UTF-8 stdout on Windows cp949 consoles
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from main import predict  # noqa: E402


def brier(probs_by_outcome: dict[str, float], outcomes: list[str], actual: str) -> float:
    """Σ(p_i − 1{i==actual})² across all submitted outcome probabilities."""
    s = 0.0
    for outcome in outcomes:
        p = probs_by_outcome.get(outcome, 0.0)
        target = 1.0 if outcome == actual else 0.0
        s += (p - target) ** 2
    return s


def score_event(event: dict, actual: str) -> dict:
    t0 = time.time()
    pred = predict(event)
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


def main(events_path: str = "events_resolved.json",
         actuals_path: str = "actuals.json",
         workers: int = 5) -> None:
    events = json.loads(Path(events_path).read_text(encoding="utf-8"))
    actuals = json.loads(Path(actuals_path).read_text(encoding="utf-8"))

    pairs = [(e, actuals[e["market_ticker"]]) for e in events
             if e["market_ticker"] in actuals]
    print(f"Scoring {len(pairs)} resolved events with {workers} parallel workers")
    print(f"Model envelope from main.py / .env\n")

    results: list[dict] = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(score_event, e, a): e["market_ticker"] for e, a in pairs}
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
        print("\nNo events scored. Check .env and main.py")
        return

    overall = sum(r["brier"] for r in results) / len(results)
    by_cat: dict[str, list[float]] = defaultdict(list)
    for r in results:
        by_cat[r["category"]].append(r["brier"])

    print(f"\n{'='*60}")
    print(f"BASELINE Brier (lower = better): {overall:.4f}")
    print(f"Reference: market consensus typically 0.15–0.22; uniform = 0.25 binary")
    print(f"Total wall clock: {total:.1f}s for {len(results)} events")
    print(f"{'='*60}\n")

    print("By category:")
    for cat, briers in sorted(by_cat.items()):
        print(f"  {cat:<20} n={len(briers):2}  avg={sum(briers)/len(briers):.4f}")

    print("\nWorst 5 (biggest Brier = biggest miss):")
    for r in sorted(results, key=lambda r: -r["brier"])[:5]:
        print(f"  {r['brier']:.4f}  [{r['n_outcomes']}-way]  {r['title']}")
        print(f"            actual={r['actual']}  predicted_actual={r['predicted'].get(r['actual'], 0.0):.3f}")

    print("\nBest 5 (smallest Brier = best calls):")
    for r in sorted(results, key=lambda r: r["brier"])[:5]:
        print(f"  {r['brier']:.4f}  {r['title']}")

    # Save full results for later analysis
    out = Path("backtest_results.json")
    out.write_text(json.dumps({"overall_brier": overall, "n": len(results),
                               "by_category": {k: sum(v)/len(v) for k, v in by_cat.items()},
                               "results": results}, indent=2), encoding="utf-8")
    print(f"\nFull results → {out}")


if __name__ == "__main__":
    args = sys.argv[1:]
    events = args[0] if len(args) > 0 else "events_resolved.json"
    actuals = args[1] if len(args) > 1 else "actuals.json"
    workers = int(args[2]) if len(args) > 2 else 5
    main(events, actuals, workers)
