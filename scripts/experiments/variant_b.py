"""Variant B — tighter binary ceiling.

For binary events (N=2), clip to [0.10, 0.90] instead of [0.05, 0.95].
Multi-outcome events use main's existing rules unchanged.

Hypothesis: the worst Brier hits in backtest_results.json come from binary
tennis matches where the LLM picked the wrong player with 0.95 confidence
(Brier = 1.805 on Perez/Lalami). A tighter cap converts those to 1.62 worst
and 0.18 for losses, trimming the tail.

Cost: same as baseline.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from main import Event, MarketProbability, Prediction, forecast as _baseline_forecast  # noqa: E402

BINARY_FLOOR = 0.10
BINARY_CEIL = 0.90


def forecast(event: Event) -> Prediction:
    pred = _baseline_forecast(event)
    if len(event.outcomes) != 2:
        return pred

    tightened: list[MarketProbability] = []
    for mp in pred.probabilities:
        p = max(BINARY_FLOOR, min(BINARY_CEIL, mp.probability))
        tightened.append(MarketProbability(market=mp.market, probability=p))

    return Prediction(
        probabilities=tightened,
        rationale=(pred.rationale + " | variant_b binary clip [0.10, 0.90]").strip(),
    )


def predict(event: dict) -> dict:
    pred = forecast(Event(**event))
    return pred.model_dump()
