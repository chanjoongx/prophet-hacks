"""Variant C — shrinkage toward uniform.

After main's forecast returns, blend each outcome's probability with the
uniform 1/N prior:  p_final = ALPHA * p_llm + (1 - ALPHA) * (1/N)
with ALPHA = 0.85. Hypothesis: cheap insurance against overconfidence.

Cost: same as baseline.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from main import Event, MarketProbability, Prediction, forecast as _baseline_forecast  # noqa: E402

ALPHA = 0.85


def forecast(event: Event) -> Prediction:
    pred = _baseline_forecast(event)
    n = max(len(event.outcomes), 1)
    u = 1.0 / n

    blended: list[MarketProbability] = []
    for mp in pred.probabilities:
        p = ALPHA * mp.probability + (1.0 - ALPHA) * u
        # safety clamp to [0,1] -- arithmetic should already keep it there
        p = max(0.0, min(1.0, p))
        blended.append(MarketProbability(market=mp.market, probability=p))

    return Prediction(
        probabilities=blended,
        rationale=(pred.rationale + f" | variant_c shrink α={ALPHA}").strip(),
    )


def predict(event: dict) -> dict:
    pred = forecast(Event(**event))
    return pred.model_dump()
