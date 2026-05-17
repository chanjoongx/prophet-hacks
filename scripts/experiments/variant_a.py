"""Variant A — self-consistency.

Call the LLM N=3 times at temperature=0.5, take the median probability per
outcome, then re-apply the existing clip rules from main. Hypothesis:
averages out single-sample noise on confidently-wrong calls.

Cost: ~3x baseline.
"""

from __future__ import annotations

import sys
from pathlib import Path
from statistics import median
from typing import Any

# Make repo root importable so we can reuse main.py wiring (env, client, prompt).
_REPO = Path(__file__).resolve().parent.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from main import (  # noqa: E402
    CLIP_MAX,
    DEFAULT_MODEL,
    SYSTEM_PROMPT,
    Event,
    MarketProbability,
    Prediction,
    _extract_json,
    _uniform_fallback,
    build_user_prompt,
    floor_for,
    get_client,
    logger,
)

N_SAMPLES = 3
TEMPERATURE = 0.5


def _raw_probs_from_json(raw: Any, outcomes: list[str]) -> dict[str, float] | None:
    """Parse one LLM call's `probabilities` field into {outcome: p} WITHOUT clipping.

    Returns None if parsing produced nothing usable for these outcomes.
    """
    by_label: dict[str, float] = {}
    if isinstance(raw, dict):
        for k, v in raw.items():
            try:
                by_label[str(k)] = float(v)
            except (TypeError, ValueError):
                continue
    elif isinstance(raw, list):
        for item in raw:
            if not isinstance(item, dict):
                continue
            label = item.get("market") or item.get("outcome") or item.get("label")
            prob = item.get("probability") or item.get("p") or item.get("prob")
            if label is None or prob is None:
                continue
            try:
                by_label[str(label)] = float(prob)
            except (TypeError, ValueError):
                continue

    out: dict[str, float] = {}
    for outcome in outcomes:
        p = by_label.get(outcome)
        if p is None:
            for label, value in by_label.items():
                if label.lower() == outcome.lower():
                    p = value
                    break
        if p is None:
            return None  # missing outcome — discard this sample
        if p > 1.0:
            p = p / 100.0
        out[outcome] = p
    return out


def _call_once(event: Event, max_tokens: int, is_search_model: bool) -> dict[str, float] | None:
    client = get_client()
    kwargs: dict[str, Any] = {
        "model": DEFAULT_MODEL,
        "max_tokens": max_tokens,
        "temperature": TEMPERATURE,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(event)},
        ],
    }
    if not is_search_model:
        kwargs["response_format"] = {"type": "json_object"}
    resp = client.chat.completions.create(**kwargs)
    text = resp.choices[0].message.content or ""
    data = _extract_json(text)
    return _raw_probs_from_json(data.get("probabilities"), event.outcomes)


def forecast(event: Event) -> Prediction:
    if not event.outcomes:
        return _uniform_fallback([], "no outcomes provided")

    is_search_model = ":online" in DEFAULT_MODEL or ":search" in DEFAULT_MODEL
    max_tokens = 1500 if is_search_model else 600

    samples: list[dict[str, float]] = []
    for i in range(N_SAMPLES):
        try:
            s = _call_once(event, max_tokens, is_search_model)
        except Exception as exc:
            logger.warning("[variant_a] sample %d failed for %s: %s", i, event.market_ticker, exc)
            continue
        if s is not None:
            samples.append(s)

    if not samples:
        return _uniform_fallback(event.outcomes, "all samples failed")

    n = max(len(event.outcomes), 1)
    floor = floor_for(n)

    probs: list[MarketProbability] = []
    for outcome in event.outcomes:
        vals = [s[outcome] for s in samples]
        m = median(vals)
        m = max(floor, min(CLIP_MAX, m))
        probs.append(MarketProbability(market=outcome, probability=m))

    return Prediction(
        probabilities=probs,
        rationale=f"variant_a self-consistency median over {len(samples)} samples",
    )


def predict(event: dict) -> dict:
    pred = forecast(Event(**event))
    return pred.model_dump()
