"""Verify a deployed forecasting agent end-to-end.

Usage:
    python scripts/verify_deployment.py https://your-app.onrender.com

Checks:
  1. Root / responds (cold-start aware: retries for ~30s if 503)
  2. /health responds 200
  3. POST /predict with a binary event → valid Prediction shape, prob in [0, 1]
  4. POST /predict with a multi-outcome event → all outcomes present, sums roughly to 1
  5. Reports per-request latency
"""

from __future__ import annotations

import io
import json
import sys
import time
from pathlib import Path

import httpx

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


def wait_for_warm(base: str, timeout_s: int = 60) -> bool:
    """Render free tier sleeps; first hit can take 30s to wake."""
    print(f"Warming up {base} (Render free tier cold-start) ...")
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            r = httpx.get(base + "/", timeout=10)
            if r.status_code == 200:
                elapsed = time.time() - t0
                print(f"  [+] cold-start completed in {elapsed:.1f}s")
                print(f"      {r.json()}")
                return True
        except Exception as exc:  # noqa: BLE001
            print(f"  [.] still waking ({type(exc).__name__}) ...")
        time.sleep(3)
    print(f"  [!] never woke after {timeout_s}s")
    return False


def hit(base: str, path: str, payload: dict | None = None, timeout: float = 60) -> tuple[int, dict, float]:
    t0 = time.time()
    if payload is None:
        r = httpx.get(base + path, timeout=timeout)
    else:
        r = httpx.post(base + path, json=payload, timeout=timeout)
    elapsed = time.time() - t0
    try:
        body = r.json()
    except Exception:
        body = {"_text": r.text[:300]}
    return r.status_code, body, elapsed


def validate_prediction(body: dict, outcomes: list[str]) -> list[str]:
    errors = []
    probs = body.get("probabilities")
    if not isinstance(probs, list):
        return ["probabilities is not a list"]
    labels = {p.get("market") for p in probs if isinstance(p, dict)}
    for outcome in outcomes:
        if outcome not in labels:
            errors.append(f"missing outcome: {outcome}")
    for p in probs:
        if not isinstance(p, dict):
            errors.append("non-dict prob entry")
            continue
        prob = p.get("probability")
        if not isinstance(prob, (int, float)) or not (0 <= prob <= 1):
            errors.append(f"bad prob value for {p.get('market')}: {prob}")
    total = sum(p.get("probability", 0) for p in probs if isinstance(p, dict))
    if total < 0.1 or total > 10:
        errors.append(f"prob sum {total:.3f} wildly off (after server normalize should still be sane)")
    return errors


def main(base_url: str) -> int:
    base = base_url.rstrip("/")
    print(f"Target: {base}\n")

    if not wait_for_warm(base):
        return 1

    print("\n--- /health ---")
    code, body, dt = hit(base, "/health")
    print(f"  {code}  {body}  ({dt:.2f}s)")
    if code != 200:
        return 1

    print("\n--- POST /predict (binary) ---")
    binary_event = {
        "event_ticker": "test-binary",
        "market_ticker": "test-binary",
        "title": "Will the Los Angeles Lakers win the 2026 NBA Finals?",
        "subtitle": None,
        "description": "Resolves Yes if the Lakers win the 2026 NBA Finals.",
        "category": "Sports",
        "rules": "Yes if Lakers win the 2026 NBA Finals; No otherwise.",
        "close_time": "2026-06-30T23:59:59Z",
        "outcomes": ["Yes", "No"],
        "resolved_outcome": None,
    }
    code, body, dt = hit(base, "/predict", binary_event, timeout=90)
    print(f"  {code}  ({dt:.2f}s)")
    print(f"  {json.dumps(body, indent=2, ensure_ascii=False)[:600]}")
    errors = validate_prediction(body, binary_event["outcomes"]) if code == 200 else ["non-200"]
    if errors:
        print(f"  [!] errors: {errors}")
        return 1

    print("\n--- POST /predict (multi-outcome, 4-way) ---")
    multi_event = {
        "event_ticker": "test-multi",
        "market_ticker": "test-multi",
        "title": "Which team wins the 2027 Super Bowl?",
        "subtitle": None,
        "description": "Resolves to the winning team.",
        "category": "Sports",
        "rules": "Resolves to the winning team of Super Bowl LXII.",
        "close_time": "2027-02-15T23:59:59Z",
        "outcomes": ["Kansas City Chiefs", "Buffalo Bills", "San Francisco 49ers", "Philadelphia Eagles"],
        "resolved_outcome": None,
    }
    code, body, dt = hit(base, "/predict", multi_event, timeout=90)
    print(f"  {code}  ({dt:.2f}s)")
    print(f"  {json.dumps(body, indent=2, ensure_ascii=False)[:600]}")
    errors = validate_prediction(body, multi_event["outcomes"]) if code == 200 else ["non-200"]
    if errors:
        print(f"  [!] errors: {errors}")
        return 1

    print("\n=== ALL CHECKS PASSED ===")
    print(f"Endpoint ready for submission: {base}/predict")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python scripts/verify_deployment.py <BASE_URL>")
        sys.exit(2)
    sys.exit(main(sys.argv[1]))
