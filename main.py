"""Prophet Hacks 2026 — Forecasting Agent.

POST /predict accepts an Event JSON (the shape produced by
`prophet forecast retrieve`) and returns a probability per outcome.

Run locally:
    uvicorn main:app --host 0.0.0.0 --port 8000

CLI smoke test:
    prophet forecast predict --events events_sports.json \\
        --agent-url http://localhost:8000/predict
"""

from __future__ import annotations

import json
import logging
import math
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Any

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from openai import OpenAI
from pydantic import BaseModel, Field

load_dotenv()

logger = logging.getLogger("agent")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

app = FastAPI(
    title="chanjoongx - Prophet Hacks Forecasting Agent",
    docs_url=None,        # disable /docs Swagger UI (hide schema from competitors)
    redoc_url=None,       # disable /redoc
    openapi_url=None,     # disable /openapi.json
)


class Event(BaseModel):
    event_ticker: str
    market_ticker: str
    title: str
    subtitle: str | None = None
    description: str | None = None
    category: str
    rules: str | None = None
    close_time: str
    outcomes: list[str] = Field(default_factory=list)
    resolved_outcome: Any | None = None


class MarketProbability(BaseModel):
    market: str
    probability: float = Field(ge=0.0, le=1.0)


class Prediction(BaseModel):
    probabilities: list[MarketProbability]
    rationale: str = ""
    # Back-compat with ai-prophet 0.1.5's CLI which calls float(result["p_yes"]).
    # The newer spec uses `probabilities`; we ship both so we satisfy whichever
    # contract the eval server is actually on.
    p_yes: float = Field(ge=0.01, le=0.99, default=0.5)


# ---------------------------------------------------------------------------
# LLM client — OpenRouter via OpenAI SDK
# ---------------------------------------------------------------------------

# Brier-aware clipping.
# - Ceiling = 0.95 (universal). Bounds penalty on confident misses.
# - Floor depends on N: binary uses 0.05, multi-outcome uses 0 because a
#   floor applied to every wrong outcome amplifies through normalization
#   and destroys mass on the actual outcome. (Verified empirically.)
CLIP_MAX = 0.95


def floor_for(n_outcomes: int) -> float:
    return 0.05 if n_outcomes <= 2 else 0.0

DEFAULT_MODEL = os.environ.get("FORECAST_MODEL", "anthropic/claude-sonnet-4")

# Multi-model ensemble for variance reduction. Each :online variant has its
# own search backend, so independent disagreement on confident-wrong calls
# pulls average toward calibration. Backtest: Brier 0.07-0.19 (solo Sonnet)
# vs 0.10 (3-way ensemble) on 26-event sample-resolved. Disable with
# USE_ENSEMBLE=0 to fall back to single-model behavior.
USE_ENSEMBLE = os.environ.get("USE_ENSEMBLE", "1") == "1"
ENSEMBLE_MODELS = [
    os.environ.get("ENSEMBLE_MODEL_1", "anthropic/claude-sonnet-4:online"),
    os.environ.get("ENSEMBLE_MODEL_2", "openai/gpt-4o:online"),
    os.environ.get("ENSEMBLE_MODEL_3", "google/gemini-2.5-flash:online"),
]

_client: OpenAI | None = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "Set OPENROUTER_API_KEY in .env (or OPENAI_API_KEY for direct OpenAI)."
            )
        base_url = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        _client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            default_headers={
                "HTTP-Referer": "https://github.com/chanjoongx",
                "X-Title": "chanjoongx-prophet-hacks",
            },
        )
    return _client


SYSTEM_PROMPT = """\
You are a calibrated probabilistic forecaster in the tradition of Tetlock's
superforecasters. Your sole task: assign a probability to each candidate outcome
for one event, scored by Brier loss (squared error). Overconfidence is punished
much more harshly than humility.

REASONING PROTOCOL (think silently, do NOT include the steps in the JSON rationale):
1. OUTSIDE VIEW FIRST. Identify the reference class for this question
   (e.g., "NBA home team in playoffs", "incumbent re-election in OECD",
   "Eurovision favorite vs dark horse"). State the base rate from that
   reference class. Use numbers when you can; honestly admit ignorance when not.
2. INSIDE VIEW. List the 2-4 strongest event-specific signals. For each, note
   direction and rough magnitude of update from the base rate.
3. DEVIL'S ADVOCATE. Spend one step arguing the opposite side. If the strongest
   counter-argument is compelling, shrink your update.
4. DATE CHECK. Compare today's date to the event close time. If the event has
   likely already resolved in your training data, treat that as a strong signal
   but not absolute certainty (records may have been amended).
5. DISTRIBUTE MASS. For multi-outcome questions, reason about each candidate
   individually. Do not default to 1/N uniform unless you truly have no signal.
   Do not lump 0.5 on the modal answer.

CALIBRATION DISCIPLINE
- Anchor your probabilities to: 5%, 10%, 20%, 35%, 50%, 65%, 80%, 90%, 95%.
  Pick the anchor closest to your true belief; avoid 0.5-clustering.
- Reserve probabilities outside [0.10, 0.90] for cases where multiple
  independent strong signals all agree.
- Reserve outside [0.05, 0.95] almost never -- a single 0.97 miss costs more
  Brier than ten 0.6 misses.
- If you genuinely do not know, shrink toward 1/N over the outcomes. Uniform
  is the Brier-minimizing prior under ignorance.
- For binary sports games with no strong edge, home-team prior ~0.58
  (NBA/MLB regular season). Start there, then adjust for actual matchup.
- Probabilities should sum to roughly 1.0; the scoring server normalizes.

PATHOLOGIES TO AVOID
- Mirroring the question's framing ("Will X happen?" biasing toward yes).
- Inventing statistics or reference-class counts you don't actually know.
- Confusing your knowledge cutoff with the event date.
- Returning labels that do not exactly match the provided outcome strings.

OUTPUT FORMAT -- return ONLY valid JSON, no prose, no markdown fences:
{
  "probabilities": [
    {"market": "<EXACT outcome label as provided>", "probability": <float in [0,1]>},
    ...
  ],
  "rationale": "<one sentence naming the reference class + base rate, one sentence on the dominant signal>"
}

Every entry in `outcomes` must appear in `probabilities` with the byte-exact
label string.
"""


def build_user_prompt(event: Event) -> str:
    today = datetime.now(tz=timezone.utc).date().isoformat()
    lines = [
        f"Today's date (UTC): {today}",
        f"Question: {event.title}",
    ]
    if event.subtitle:
        lines.append(f"Subtitle: {event.subtitle}")
    if event.description and event.description != event.rules:
        lines.append(f"Description: {event.description}")
    if event.rules:
        lines.append(f"Resolution rules: {event.rules}")
    lines.append(f"Category: {event.category}")
    lines.append(f"Event closes: {event.close_time}")
    lines.append("")
    lines.append("Candidate outcomes (return a probability for each, using the EXACT label):")
    for o in event.outcomes:
        lines.append(f"  - {o}")
    lines.append("")
    lines.append("Reason silently through the protocol, then return JSON only.")
    return "\n".join(lines)


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
    if text.endswith("```"):
        text = text.rsplit("```", 1)[0]
    return text.strip()


def _extract_json(text: str) -> Any:
    """Parse JSON from text that may have prose, citations, or markdown around it."""
    text = _strip_fences(text)
    # Fast path
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Find the first {...} block and try parsing greedily, then shrinking
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise json.JSONDecodeError("no JSON object found", text, 0)
    # Try the whole span first, then progressively shrink end
    span = text[start : end + 1]
    try:
        return json.loads(span)
    except json.JSONDecodeError:
        # Walk backwards finding balanced }
        depth = 0
        in_str = False
        esc = False
        for i, ch in enumerate(span):
            if esc:
                esc = False
                continue
            if ch == "\\":
                esc = True
                continue
            if ch == '"':
                in_str = not in_str
                continue
            if in_str:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return json.loads(span[: i + 1])
        raise json.JSONDecodeError("unbalanced braces", text, 0)


def _coerce_probabilities(raw: Any, outcomes: list[str]) -> list[MarketProbability]:
    """Accept dict-shaped or list-shaped LLM output and snap to our outcome labels."""
    by_label: dict[str, float] = {}

    if isinstance(raw, dict):
        for k, v in raw.items():
            try:
                fv = float(v)
            except (TypeError, ValueError):
                continue
            if math.isfinite(fv):
                by_label[str(k)] = fv
    elif isinstance(raw, list):
        for item in raw:
            if not isinstance(item, dict):
                continue
            label = item.get("market") or item.get("outcome") or item.get("label")
            prob = item.get("probability") or item.get("p") or item.get("prob")
            if label is None or prob is None:
                continue
            try:
                fv = float(prob)
            except (TypeError, ValueError):
                continue
            if math.isfinite(fv):
                by_label[str(label)] = fv

    # Snap to canonical outcome labels; fall back to uniform for anything missing.
    n = max(len(outcomes), 1)
    floor = floor_for(n)
    result: list[MarketProbability] = []
    for outcome in outcomes:
        p = by_label.get(outcome)
        if p is None:
            # case-insensitive + whitespace-tolerant fallback
            outcome_key = outcome.strip().lower()
            for label, value in by_label.items():
                if label.strip().lower() == outcome_key:
                    p = value
                    break
        if p is None:
            p = 1.0 / n  # uniform fallback for missing outcome
        # If LLM accidentally returned percent
        if p > 1.0:
            p = p / 100.0
        p = max(floor, min(CLIP_MAX, p))
        result.append(MarketProbability(market=outcome, probability=p))
    return result


def _compute_p_yes(probs: list[MarketProbability]) -> float:
    """Derive a single binary p_yes for the v0.1.5 CLI back-compat field.

    Prefers a "Yes" outcome if present (case-insensitive), else the first
    outcome's probability. Clamped to [0.01, 0.99] to satisfy the old
    schema's Field(ge=0.01, le=0.99) constraint.
    """
    if not probs:
        return 0.5
    yes_p = next((p.probability for p in probs if p.market.strip().lower() == "yes"), None)
    if yes_p is None:
        yes_p = probs[0].probability
    return max(0.01, min(0.99, float(yes_p)))


def _uniform_fallback(outcomes: list[str], reason: str) -> Prediction:
    n = max(len(outcomes), 1)
    probs = [MarketProbability(market=o, probability=1.0 / n) for o in outcomes]
    return Prediction(
        probabilities=probs,
        rationale=f"Fallback uniform: {reason}",
        p_yes=_compute_p_yes(probs),
    )


# Per-model timeout to defend against the prophethacks.com submit-endpoint
# check (which times out at 25s, NOT 10min). If one ensemble model is slow,
# we cap its wait and use whichever models finished in time. Eval timeout is
# still 10min so this is purely a robustness floor.
PER_MODEL_TIMEOUT_S = float(os.environ.get("PER_MODEL_TIMEOUT_S", "20"))


def _call_single_model(model: str, event: Event) -> tuple[list[MarketProbability] | None, str]:
    """One LLM call. Returns (coerced probabilities, rationale) or (None, error)."""
    is_search_model = ":online" in model or ":search" in model
    max_tokens = 1500 if is_search_model else 1200
    try:
        client = get_client()
        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": 0.2,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(event)},
            ],
        }
        # response_format=json_object is supported by most non-search OpenRouter
        # models. Search models put citations outside the JSON so we skip it.
        if not is_search_model:
            kwargs["response_format"] = {"type": "json_object"}
        # Per-request timeout via OpenAI SDK's with_options. If this single
        # model is slow, raise APITimeoutError which our except below catches.
        resp = client.with_options(timeout=PER_MODEL_TIMEOUT_S).chat.completions.create(**kwargs)
        text = resp.choices[0].message.content or ""
        data = _extract_json(text)
        if isinstance(data, list):
            raw_probs: Any = data
            rationale = ""
        elif isinstance(data, dict):
            raw_probs = data.get("probabilities")
            rationale = str(data.get("rationale", ""))[:200]
        else:
            return None, f"unexpected JSON type {type(data).__name__}"
        probs = _coerce_probabilities(raw_probs, event.outcomes)
        return probs, rationale
    except Exception as exc:
        return None, f"{type(exc).__name__}: {str(exc)[:120]}"


def forecast(event: Event) -> Prediction:
    if not event.outcomes:
        return _uniform_fallback([], "no outcomes provided")

    models = ENSEMBLE_MODELS if USE_ENSEMBLE else [DEFAULT_MODEL]

    samples: list[list[MarketProbability]] = []
    rationales: list[str] = []
    failures: list[str] = []

    if len(models) == 1:
        # Single-model path — synchronous to keep things simple
        probs, rat_or_err = _call_single_model(models[0], event)
        if probs is not None:
            samples.append(probs)
            rationales.append(rat_or_err)
        else:
            failures.append(f"{models[0]}: {rat_or_err}")
    else:
        # Parallel fan-out; wall time ~= slowest model, not sum
        with ThreadPoolExecutor(max_workers=len(models)) as ex:
            future_to_model = {ex.submit(_call_single_model, m, event): m for m in models}
            for fut in as_completed(future_to_model):
                model = future_to_model[fut]
                try:
                    probs, rat_or_err = fut.result()
                except Exception as exc:
                    failures.append(f"{model}: {type(exc).__name__}")
                    continue
                if probs is not None:
                    samples.append(probs)
                    rationales.append(rat_or_err)
                else:
                    failures.append(f"{model}: {rat_or_err}")

    if not samples:
        logger.warning("All models failed for %s: %s", event.market_ticker, failures)
        return _uniform_fallback(event.outcomes, f"all models failed: {failures[:3]}")

    # Mean per outcome across surviving samples
    n = max(len(event.outcomes), 1)
    floor = floor_for(n)
    averaged: list[MarketProbability] = []
    for i, outcome in enumerate(event.outcomes):
        vals = [s[i].probability for s in samples if i < len(s)]
        if not vals:
            avg = 1.0 / n
        else:
            avg = sum(vals) / len(vals)
        if not math.isfinite(avg):
            avg = 1.0 / n
        avg = max(floor, min(CLIP_MAX, avg))
        averaged.append(MarketProbability(market=outcome, probability=avg))

    rationale = (
        f"ensemble mean ({len(samples)}/{len(models)} models); "
        + (rationales[0] if rationales else "")
    )[:500]
    return Prediction(
        probabilities=averaged,
        rationale=rationale,
        p_yes=_compute_p_yes(averaged),
    )


# ---------------------------------------------------------------------------
# `prophet forecast predict --local main` entrypoint
# ---------------------------------------------------------------------------

def predict(event: dict) -> dict:
    pred = forecast(Event(**event))
    return pred.model_dump()


# ---------------------------------------------------------------------------
# HTTP endpoint (this is what the eval server hits)
# ---------------------------------------------------------------------------

@app.get("/")
def root() -> dict[str, str]:
    return {"status": "ok", "agent": "chanjoongx"}


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=Prediction)
def predict_endpoint(event: Event) -> Prediction:
    logger.info("Predict %s | %s outcomes | %s", event.market_ticker, len(event.outcomes), event.title)
    return forecast(event)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
