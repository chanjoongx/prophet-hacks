"""Variant D — multi-model ensemble.

Call Sonnet, GPT-4o, and Gemini-Flash (all :online search variants) once each
at temperature=0.2, then average per-outcome probabilities and re-apply main's
clip rules. Hypothesis: the dominant failure mode in production backtests is a
single confident-wrong call when one model's search latches onto a wrong
source (Perez/Lalami's everygame.eu, Colombia Senate, SCOTUS vote split).
Different models have different search backends and different attention to
sources, so disagreement on disasters is more likely than agreement.

Why average vs median: with only 3 samples and a bimodal "two right, one
wrong" pattern, median rounds to the majority, which is exactly what we want
on disasters — but average is more robust when two models give 0.7/0.8 and
one gives 0.3 (median would discard the 0.3 entirely; average pulls toward
calibration). Tiny difference; mean is the safer default.

Cost: ~3x baseline LLM cost (one call per model). Latency: parallel via
ThreadPoolExecutor so wall time ~= slowest model.

Risk: if any one model is much worse than Sonnet on the workload, averaging
would drag us down. Mitigation: fall back to Sonnet-only when 2+ models fail.
"""

from __future__ import annotations

import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parent.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from main import (  # noqa: E402
    CLIP_MAX,
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

# All :online so they actually search the live web. If a model's :online
# variant is unavailable on OpenRouter, swap it for the plain id.
ENSEMBLE_MODELS = [
    os.environ.get("ENSEMBLE_MODEL_1", "anthropic/claude-sonnet-4:online"),
    os.environ.get("ENSEMBLE_MODEL_2", "openai/gpt-4o:online"),
    os.environ.get("ENSEMBLE_MODEL_3", "google/gemini-2.5-flash:online"),
]

TEMPERATURE = 0.2


def _raw_probs_from_json(raw: Any, outcomes: list[str]) -> dict[str, float] | None:
    """Reuse main's lenient coercion so missing outcomes get 1/N rather than
    discarding the whole model's contribution."""
    from main import _coerce_probabilities
    coerced = _coerce_probabilities(raw, outcomes)
    if not coerced:
        return None
    return {p.market: p.probability for p in coerced}


def _call_model(model: str, event: Event) -> dict[str, float] | None:
    client = get_client()
    is_search_model = ":online" in model or ":search" in model
    max_tokens = 1500 if is_search_model else 1200
    kwargs: dict[str, Any] = {
        "model": model,
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
    if isinstance(data, list):
        raw = data
    elif isinstance(data, dict):
        raw = data.get("probabilities")
    else:
        return None
    return _raw_probs_from_json(raw, event.outcomes)


def forecast(event: Event) -> Prediction:
    if not event.outcomes:
        return _uniform_fallback([], "no outcomes provided")

    samples: list[dict[str, float]] = []
    failures: list[str] = []
    # Parallel fan-out so wall time = slowest model, not sum of all three.
    with ThreadPoolExecutor(max_workers=len(ENSEMBLE_MODELS)) as ex:
        futures = {ex.submit(_call_model, m, event): m for m in ENSEMBLE_MODELS}
        for fut in as_completed(futures):
            model = futures[fut]
            try:
                s = fut.result()
            except Exception as exc:
                failures.append(f"{model}:{type(exc).__name__}")
                logger.warning("[variant_d] %s failed for %s: %s", model, event.market_ticker, exc)
                continue
            if s is not None:
                samples.append(s)
            else:
                failures.append(f"{model}:parse")

    if not samples:
        return _uniform_fallback(event.outcomes, f"all ensemble models failed: {failures}")

    n = max(len(event.outcomes), 1)
    floor = floor_for(n)
    probs: list[MarketProbability] = []
    for outcome in event.outcomes:
        vals = [s[outcome] for s in samples]
        avg = sum(vals) / len(vals)
        avg = max(floor, min(CLIP_MAX, avg))
        probs.append(MarketProbability(market=outcome, probability=avg))

    return Prediction(
        probabilities=probs,
        rationale=f"variant_d ensemble mean over {len(samples)}/{len(ENSEMBLE_MODELS)} models"
                  + (f"; failures={failures}" if failures else ""),
    )


def predict(event: dict) -> dict:
    pred = forecast(Event(**event))
    return pred.model_dump()
