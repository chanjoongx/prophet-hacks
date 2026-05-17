# Calibrated Multi-Outcome Forecasting Agent

A calibration-first forecasting agent for the Prophet Hacks 2026 AI Forecasting Track — built solo, deployed end-to-end, and designed around the actual scoring function rather than the demo template.

## Inspiration

Prediction markets like Kalshi are one of the few places where being a *little* better calibrated than the crowd is directly, quantitatively valuable. The AI Forecasting Track puts that into a tight loop: an event comes in, you return probabilities, Brier score grades you. I wanted to see how far a small, focused agent could get against that loop in a weekend — not by being clever, but by being honest about uncertainty and by exploiting every angle of *web-grounded* reasoning at once.

## What it does

The agent exposes a single `POST /predict` endpoint that consumes the official Prophet `Event` schema (the same payload returned by `prophet forecast retrieve`) and responds with a probability per outcome that respects the eval contract: every market label appears, every probability in `[0, 1]`, total ~1.0 so server normalization is a no-op.

It works for both binary markets (Yes/No) and arbitrary multi-outcome markets (e.g., the 30-way NHL Calder Trophy race) using the same code path. The architecture's main job is to turn an event into a calibrated distribution and never let a transient provider hiccup return a 5xx.

Final production Brier on the public `sample-resolved` set: **~0.04 excluding one search-noise outlier, ~0.09 including it** — vs market consensus baseline ~0.15–0.22.

## How we built it

The architecture is intentionally boring in the structure and aggressive in the inference. Most lines are spent on parsing and fallbacks; the interesting decisions are the ensemble shape and the calibration prompt.

- **FastAPI + Uvicorn** server with a single `POST /predict` endpoint and a Pydantic model matching the official Event contract. `/docs`, `/redoc`, and `/openapi.json` are disabled so schema choices don't leak to competing teams.
- **4-way `:online` ensemble through OpenRouter:** Claude Sonnet 4 × 2 (self-consistency), GPT-4o, and Gemini-2.5-Flash, each with the `:online` suffix and `max_results=10` so each call gets ~10 Exa-backed search results before reasoning. Independent search backends (Brave/Tavily, Bing, Google) are the actual diversity here, not the model names — when one model's first search result is misleading, another model's different search backend is the corrective signal.
- **Hybrid residual-mass coercion:** for outcomes the LLM didn't explicitly mention in its JSON, we fill with `max(true_residual_mass, 0.5/N)` — a humility floor at half the uniform fallback. Empirically this beat both the naive 1/N fill (which inflated sums and got deflated under server normalization) and pure residual mass (which amplified confidently-wrong calls into outsized Brier hits).
- **Asymmetric N-aware clipping:** ceiling `0.95` universal, floor `0.05` for binary and `0` for multi-outcome. A floor on every wrong outcome in a 30-way race destroys the actual outcome's mass after normalization.
- **3-tier independent-infrastructure fallback** for when OpenRouter is fully down: Anthropic Sonnet direct → OpenAI GPT-4o direct → uniform `1/N`. Tested empirically with mocked outages. (Direct providers are NOT in the active ensemble path — when I tried that, no-search models contributed mediocre guesses on recent events that diluted the search-grounded correct answers. Disabled by `USE_DIRECT_PROVIDERS=0`; keys still active in the fallback chain.)
- **Tetlock-style calibration prompt:** Outside view first → inside view → devil's advocate → date check → distribute mass. Anchors to specific probability values (5%, 10%, 20%, …, 95%) to break 0.5-clustering. Today's UTC date injected into every prompt because the single highest-ROI prompt change in the Halawi paper was telling the model what date it is.
- **Defensive output parser:** balanced-brace JSON extractor that ignores markdown citations the `:online` responses append, case-insensitive + whitespace-tolerant outcome label matching, NaN/Inf filtering, percentage auto-detection. Every error path returns a uniform-fallback Prediction so the endpoint never 5xx's — missed events would tank the completion-rate factor in the leaderboard formula.
- **Per-model 20s timeout** to defend against prophethacks.com's submit-endpoint forecast-check (25s budget, NOT the 10-min eval budget). Bounds tail latency even when one model goes slow.
- **Back-compat `p_yes` field** in the response alongside `probabilities` so the agent satisfies both the multi-outcome spec from the developer docs AND the older `ai-prophet 0.1.5` CLI which calls `float(result["p_yes"])`. Extras aren't prohibited by either contract.
- **Local Brier backtest harness** replaying `sample-resolved` against the running server. Every prompt and clipping change got a Brier delta before it shipped. Several "obviously good" ideas (uniform shrinkage, tighter binary ceiling, no-search direct ensemble members) lost in backtest and got reverted — the harness was load-bearing.
- **Render free tier** for deployment with auto-deploy on git push. `Dockerfile` and `evaluate.sh` included for organizer reproducibility per the rules.

## Challenges we ran into

- **The schema docs contradict themselves.** Quick Start says "probabilities normalized before scoring", Custom Agent says "should sum to 1", participant guide says "OpenAI-compatible endpoint". An organizer clarified on Discord that it's loose phrasing — actual contract is the per-event POST with the explicit `{probabilities: [...]}` response. I built to that and added the `p_yes` field defensively.
- **Multi-outcome handling exposed every brittle assumption.** Going from "return a float" to "return a normalized distribution over N labels the model may not have explicitly named in its output" is where most bugs lived. The hybrid residual-mass logic was rewritten three times before backtest agreed it was right.
- **Ensemble doesn't fix systematic search misinformation.** When `:online` lands the LLM on a wrong source, additional samples of the same model trust the same source. The genuine variance reduction came from *different search backends*, not more samples of one model — that insight reshaped the ensemble composition midway through the build.
- **Calibration vs hedging tradeoff isn't theoretical.** Pure residual-mass coercion (sum stays ~1.0, server normalization is a no-op) won big on confident-correct cases but amplified the few confident-wrong cases into Brier 1.7+ disasters. The hybrid floor at `0.5/N` was the compromise — picked by backtest, not by feel.
- **Solo time pressure.** No teammate to bounce prompts off, no second pair of eyes on the parser. Discipline came from the Brier loop — if a change didn't improve the score on `sample-resolved`, it didn't ship. Several theoretically clever ideas reverted because the data said so.

## Accomplishments that we're proud of

- Multi-outcome native from the first commit, not bolted on.
- 4-way search ensemble with `max_results=10` and a hybrid residual-mass coercion that's mathematically correct and empirically the best of three variants I tested.
- 3-tier provider-redundant fallback chain (OpenRouter ensemble → Anthropic direct → OpenAI direct → uniform). The endpoint has not returned a 5xx across all backtests.
- Every architectural decision is backed by a backtest Brier delta — including the decisions to *not* ship features that seemed obviously useful but lost in the data.
- Full path from `git clone` to a live Render URL with three-provider redundancy in under 24 hours, solo.

## What we learned

- Brier rewards humility surprisingly hard. The math on `(p - outcome)^2` means a 0.95 that turns out wrong costs you ~10× what a 0.7 that turns out wrong does. Most "improvements" to a forecasting agent are actually just learning to stop overclaiming.
- Search backend diversity > model diversity. Different LLMs reading the same wrong source agree on the same wrong answer; different search engines surface different sources, and that's where the ensemble actually pays its bills.
- The fastest way to lose Brier is to forget the date. Injecting today's UTC date into the user prompt was a one-line change that materially reduced "model confused training cutoff with event date" errors.
- A tight, local evaluation loop is worth more than any individual prompt trick. Without `sample-resolved` to measure against, every change is a guess.

## What's next for the project

- **Category-specific calibration**, especially for "vote count" style events (SCOTUS, US Senate confirmations) where `:online` search consistently lands on the wrong tally and the LLMs get confidently wrong together. Constrained-output decoding with a domain prior might be the fix.
- **Cross-provider Opus tier** for hard events: a Haiku-class classifier decides "is this event subtle enough to need Opus?" and routes accordingly. Cost-targeted, not blanket.
- **RAG over historical Kalshi data** to give the model an actual empirical base rate for similar past markets instead of asking it to guess one.
- **Devil's-advocate second pass** triggered only when the ensemble lands on `max(p) ≥ 0.85`. Cheap insurance against confident-wrong on rare categories.

## Built with

Python 3.12, FastAPI, Uvicorn, Pydantic, OpenRouter, Anthropic SDK, OpenAI SDK, Claude Sonnet 4, GPT-4o, Gemini-2.5-Flash, Render.

## Try it yourself

```bash
git clone https://github.com/chanjoongx/prophet-hacks
cd prophet-hacks
pip install -r requirements.txt
cp .env.example .env   # add your OPENROUTER_API_KEY (and optionally ANTHROPIC_API_KEY, OPENAI_FALLBACK_API_KEY)
uvicorn main:app --reload
```

Then point the Prophet CLI at `http://localhost:8000/predict`, or hit the deployed Render URL (`https://chanjoongx-prophet-hacks.onrender.com/predict`) directly. Run `python scripts/backtest.py` to replay `sample-resolved` end-to-end and see Brier. `bash evaluate.sh` does the full round-trip in one shot per the hackathon submission rules.

Source: https://github.com/chanjoongx/prophet-hacks
