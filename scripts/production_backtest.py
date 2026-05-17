"""Production backtest — POST resolved events to the LIVE deployed URL.

Confirms the deployed service produces the same Brier as the local backtest.
If production Brier != local Brier (within noise), something is misconfigured
(env var mismatch, wrong commit deployed, model fallback, etc.).

Usage:
    python scripts/production_backtest.py https://chanjoongx-prophet-hacks.onrender.com
"""

from __future__ import annotations

import io
import json
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


def brier(probs_by_outcome: dict[str, float], outcomes: list[str], actual: str) -> float:
    s = 0.0
    for outcome in outcomes:
        p = probs_by_outcome.get(outcome, 0.0)
        target = 1.0 if outcome == actual else 0.0
        s += (p - target) ** 2
    return s


def score_event(base_url: str, event: dict, actual: str) -> dict:
    t0 = time.time()
    try:
        r = httpx.post(base_url + "/predict", json=event, timeout=120)
        r.raise_for_status()
        pred = r.json()
        probs = {p["market"]: float(p["probability"]) for p in pred.get("probabilities", [])}
        b = brier(probs, event["outcomes"], actual)
        return {
            "ticker": event["market_ticker"],
            "title": event["title"][:70],
            "category": event["category"],
            "n_outcomes": len(event["outcomes"]),
            "actual": actual,
            "actual_prob": probs.get(actual, 0.0),
            "brier": b,
            "elapsed_s": time.time() - t0,
            "status": "ok",
        }
    except Exception as exc:
        return {
            "ticker": event["market_ticker"],
            "title": event["title"][:70],
            "category": event["category"],
            "brier": 1.0,
            "elapsed_s": time.time() - t0,
            "status": f"ERROR: {type(exc).__name__}: {exc}"[:120],
        }


def main(base_url: str, events_path: str = "events_resolved.json",
         actuals_path: str = "actuals.json", workers: int = 3) -> None:
    base = base_url.rstrip("/")
    events = json.loads(Path(events_path).read_text(encoding="utf-8"))
    actuals = json.loads(Path(actuals_path).read_text(encoding="utf-8"))
    pairs = [(e, actuals[e["market_ticker"]]) for e in events
             if e["market_ticker"] in actuals]

    print(f"Target: {base}")
    print(f"Scoring {len(pairs)} resolved events, {workers} workers")
    print(f"(first request may be slow due to Render cold-start)\n")

    # Warm up
    try:
        httpx.get(base + "/", timeout=60)
    except Exception:
        pass

    results: list[dict] = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(score_event, base, e, a): e["market_ticker"]
                   for e, a in pairs}
        for i, fut in enumerate(as_completed(futures), 1):
            r = fut.result()
            results.append(r)
            tag = "+" if r["brier"] < 0.25 else "-"
            err = "" if r.get("status") == "ok" else f"  [{r.get('status')}]"
            print(f"  [{i}/{len(pairs)}] {tag} {r['ticker']}  Brier={r['brier']:.4f}  ({r['elapsed_s']:.1f}s){err}")

    elapsed = time.time() - t0
    ok = [r for r in results if r.get("status") == "ok"]
    overall = sum(r["brier"] for r in ok) / len(ok) if ok else 0.0
    errors = [r for r in results if r.get("status") != "ok"]

    print(f"\n{'='*60}")
    print(f"PRODUCTION Brier: {overall:.4f}  (lower = better)")
    print(f"Reference local Brier: ~0.168;  market consensus: 0.15-0.22")
    print(f"OK: {len(ok)}/{len(results)}  errors: {len(errors)}")
    print(f"Wall: {elapsed:.1f}s")
    print(f"{'='*60}")

    by_cat: dict[str, list[float]] = defaultdict(list)
    for r in ok:
        by_cat[r["category"]].append(r["brier"])
    print("\nBy category:")
    for cat, briers in sorted(by_cat.items()):
        print(f"  {cat:<18} n={len(briers):2}  avg={sum(briers)/len(briers):.4f}")

    if errors:
        print("\nERRORS:")
        for r in errors:
            print(f"  {r['ticker']:30}  {r.get('status', '?')}")

    out_path = Path("production_backtest_results.json")
    out_path.write_text(
        json.dumps({"base_url": base, "overall_brier": overall,
                    "ok": len(ok), "errors": len(errors),
                    "by_category": {k: sum(v) / len(v) for k, v in by_cat.items()},
                    "results": results}, indent=2),
        encoding="utf-8",
    )
    print(f"\nFull results -> {out_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/production_backtest.py <BASE_URL>")
        sys.exit(2)
    main(sys.argv[1])
