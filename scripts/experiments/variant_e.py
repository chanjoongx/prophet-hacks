"""Variant E — adversarial second pass (devil's advocate shrinkage).

Two-stage:
  1. Baseline call (Sonnet :online, temp 0.2) -> initial probabilities.
  2. If max(prob) >= 0.85 on any outcome, fire a second call that explicitly
     asks the model to argue the OPPOSITE side and re-estimate. We then
     compare:
        - if the adversarial call agrees within 0.10 -> keep stage-1 confident
          (we have corroboration, do not shrink).
        - if it disagrees materially (top outcome drops by >= 0.15) -> blend
          the two: p_final = 0.5*p1 + 0.5*p2. This is the only path that
          shrinks; it specifically targets cases where the model can construct
          a real counter-narrative.

Hypothesis: the dominant failure mode is one confident-wrong call per ~26
events (Perez/Lalami: 1.805 Brier from 0.05 on actual). If devil's-advocate
surfaces a real counter-signal on a meaningful fraction of disasters, the
asymmetry is favorable:
    - Disaster avoided: 0.95 -> 0.50 -> Brier 1.805 -> 0.50 saves 1.305.
    - False alarm on correct call: 0.95 -> 0.50 -> Brier 0.005 -> 0.50
      costs 0.495.
    So a hit-rate of >27% on disasters vs <100% on correct calls is +EV.

Only triggers the second call on high-confidence forecasts, so cost is
roughly 1.5-2x baseline (most events are confidently called by stage 1).

Risk: if devil's-advocate is reliably bullshit-generating (the LLM can always
manufacture a plausible counter-argument even for correct calls), this
degrades us. Test on backtest first.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

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

CONFIDENCE_TRIGGER = float(os.environ.get("VARIANT_E_TRIGGER", "0.85"))
DISAGREEMENT_THRESHOLD = float(os.environ.get("VARIANT_E_DISAGREE", "0.15"))
BLEND_WEIGHT = float(os.environ.get("VARIANT_E_BLEND", "0.5"))

DEVILS_ADVOCATE_SYSTEM = """\
You are a skeptical forecaster doing a SECOND-PASS audit on a prior forecast.
You will be given a question and an INITIAL probability assignment from a
previous reasoner. Your job:

1. Argue the strongest counter-case to the initial top pick. Look specifically
   for: source-quality issues (one site might be wrong), date confusion
   (training-data cutoff vs event date), reference-class shifts, ambiguous
   resolution rules.
2. If after honest adversarial reasoning the initial top pick still seems
   right, say so AND keep the high probability. DO NOT shrink for the sake of
   shrinking — manufactured doubt is worse than honest confidence.
3. If you uncover a real counter-signal, redistribute probability mass.

Return ONLY valid JSON, no prose, no markdown fences:
{
  "probabilities": [
    {"market": "<EXACT outcome label>", "probability": <float in [0,1]>},
    ...
  ],
  "rationale": "<one sentence on the strongest counter-argument; one sentence on whether it survived scrutiny>"
}
"""


def _raw_probs_from_json(raw: Any, outcomes: list[str]) -> dict[str, float] | None:
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
            return None
        if p > 1.0:
            p = p / 100.0
        out[outcome] = p
    return out


def _stage1(event: Event) -> tuple[dict[str, float], str] | None:
    client = get_client()
    is_search_model = ":online" in DEFAULT_MODEL or ":search" in DEFAULT_MODEL
    max_tokens = 1500 if is_search_model else 1200
    kwargs: dict[str, Any] = {
        "model": DEFAULT_MODEL,
        "max_tokens": max_tokens,
        "temperature": 0.2,
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
    if isinstance(data, dict):
        raw = data.get("probabilities")
        rationale = str(data.get("rationale", ""))[:500]
    elif isinstance(data, list):
        raw = data
        rationale = ""
    else:
        return None
    parsed = _raw_probs_from_json(raw, event.outcomes)
    if parsed is None:
        return None
    return parsed, rationale


def _stage2_devils_advocate(event: Event, stage1: dict[str, float],
                            stage1_rationale: str) -> dict[str, float] | None:
    client = get_client()
    is_search_model = ":online" in DEFAULT_MODEL or ":search" in DEFAULT_MODEL
    max_tokens = 1500 if is_search_model else 1200

    stage1_str = "\n".join(f"  - {o}: {stage1[o]:.3f}" for o in event.outcomes)
    audit_user = (
        f"{build_user_prompt(event)}\n\n"
        f"INITIAL PROBABILITY ASSIGNMENT (from a previous reasoner):\n"
        f"{stage1_str}\n"
        f"Initial rationale: {stage1_rationale}\n\n"
        f"Audit this. Return JSON only."
    )
    kwargs: dict[str, Any] = {
        "model": DEFAULT_MODEL,
        "max_tokens": max_tokens,
        "temperature": 0.3,
        "messages": [
            {"role": "system", "content": DEVILS_ADVOCATE_SYSTEM},
            {"role": "user", "content": audit_user},
        ],
    }
    if not is_search_model:
        kwargs["response_format"] = {"type": "json_object"}
    resp = client.chat.completions.create(**kwargs)
    text = resp.choices[0].message.content or ""
    data = _extract_json(text)
    if isinstance(data, dict):
        raw = data.get("probabilities")
    elif isinstance(data, list):
        raw = data
    else:
        return None
    return _raw_probs_from_json(raw, event.outcomes)


def forecast(event: Event) -> Prediction:
    if not event.outcomes:
        return _uniform_fallback([], "no outcomes provided")

    n = max(len(event.outcomes), 1)
    floor = floor_for(n)

    try:
        s1 = _stage1(event)
    except Exception as exc:
        logger.warning("[variant_e] stage1 failed for %s: %s", event.market_ticker, exc)
        return _uniform_fallback(event.outcomes, f"stage1 error: {type(exc).__name__}")
    if s1 is None:
        return _uniform_fallback(event.outcomes, "stage1 parse failure")
    p1, r1 = s1

    top_outcome = max(event.outcomes, key=lambda o: p1[o])
    top_prob = p1[top_outcome]

    # Cheap path: not confident enough to bother with the audit.
    if top_prob < CONFIDENCE_TRIGGER:
        probs = []
        for o in event.outcomes:
            v = max(floor, min(CLIP_MAX, p1[o]))
            probs.append(MarketProbability(market=o, probability=v))
        return Prediction(probabilities=probs, rationale=r1)

    # Stage 2: adversarial audit.
    try:
        p2 = _stage2_devils_advocate(event, p1, r1)
    except Exception as exc:
        logger.warning("[variant_e] stage2 failed for %s: %s", event.market_ticker, exc)
        p2 = None

    final: dict[str, float]
    rationale: str
    if p2 is None:
        final = p1
        rationale = r1 + " | (devil's-advocate failed; keeping stage 1)"
    else:
        drop = p1[top_outcome] - p2[top_outcome]
        if drop >= DISAGREEMENT_THRESHOLD:
            # Material disagreement: blend.
            final = {o: BLEND_WEIGHT * p1[o] + (1 - BLEND_WEIGHT) * p2[o] for o in event.outcomes}
            rationale = (
                f"stage1={p1[top_outcome]:.2f} -> stage2={p2[top_outcome]:.2f} on {top_outcome}; "
                f"blended {BLEND_WEIGHT:.2f}/{1-BLEND_WEIGHT:.2f}"
            )
        else:
            final = p1
            rationale = f"devil's-advocate corroborated stage1 ({p2[top_outcome]:.2f} vs {p1[top_outcome]:.2f})"

    probs: list[MarketProbability] = []
    for o in event.outcomes:
        v = max(floor, min(CLIP_MAX, final[o]))
        probs.append(MarketProbability(market=o, probability=v))
    return Prediction(probabilities=probs, rationale=rationale)


def predict(event: dict) -> dict:
    pred = forecast(Event(**event))
    return pred.model_dump()
