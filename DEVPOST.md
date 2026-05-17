# Calibrated Multi-Outcome Forecasting Agent

A calibration-first forecasting agent for the Prophet Hacks 2026 AI Forecasting Track — built solo, deployed end-to-end, and designed around the actual scoring function rather than the demo template.

## Inspiration

Prediction markets like Kalshi are one of the few places where being a *little* better calibrated than the crowd is directly, quantitatively valuable. The AI Forecasting Track puts that into a tight loop: an event comes in, you return probabilities, Brier score grades you. I wanted to see how far a small, focused agent could get against that loop in a weekend — not by being clever, but by being honest about uncertainty.

## What it does

The agent exposes a single `POST /predict` endpoint that consumes the official Prophet `Event` schema (the same payload returned by `prophet forecast retrieve`) and responds with a probability distribution over the event's outcomes that sums to 1.0.

It works for both binary markets (Yes/No) and arbitrary multi-outcome markets (e.g., "Who wins the NBA Finals?" with 16 teams) using the same code path. Probabilities are clipped with an asymmetric, N-aware bound — ceiling 0.95 universally, floor 0.05 for binary and 0 for multi-outcome — because applying a floor to every wrong outcome in a 30-way race destroys the mass on the actual outcome after server-side normalization.

## How we built it

The architecture is intentionally boring so the interesting part — the prompt and the parsing — gets all the attention.

- **FastAPI + Uvicorn** server with a single endpoint and a Pydantic model matching the official Event contract.
- **OpenRouter** as the LLM gateway instead of hardcoding the Anthropic SDK. The model id lives in `.env`, so swapping Claude Sonnet 4 for any other model is a one-line change — useful for ablations and for routing different event categories to different models later.
- **`:online` suffix for web-grounded forecasts.** Default model is `anthropic/claude-sonnet-4:online`, which routes through OpenRouter's Exa-backed search before answering. This was the single largest Brier improvement — events whose resolution lies past the LLM's knowledge cutoff (most of them, in a live forecasting setting) drop from near-uniform (~0.95 Brier on 30-way) to confident-correct (~0.03) once the model can read the news.
- **Calibration-focused system prompt** that explicitly asks the model to reason about base rates, pick a reference class, weigh evidence directionally, and *then* commit to numbers. The output contract is a JSON object mapping outcome labels to probabilities.
- **Defensive output parser** that accepts either a `{"label": p}` dict or a list of `{label, probability}` objects, matches labels case-insensitively against the event's declared outcomes, handles missing outcomes by filling with the residual mass, and falls back to a uniform distribution on total parse failure. Includes a balanced-brace extractor for search-enabled models that append citations after the JSON. This was load-bearing.
- **Local backtest harness** in `scripts/` that replays the `sample-resolved` dataset against the running server and prints per-event and aggregate Brier. This was the actual development tool — every prompt change got a Brier delta before it shipped.
- **Render free tier** for deployment. One service, one env var, one URL.

What makes it different from the stock example agent: the example returns a single `p_yes` float and is structurally binary-only. This agent is multi-outcome native, which matters because a meaningful fraction of Kalshi-style events aren't binary.

## Challenges we ran into

- **Schema ambiguity.** The public docs and the actual `prophet forecast retrieve` payload didn't agree on every field. The fix was to treat the live CLI output as ground truth and make Pydantic permissive about extras.
- **Multi-outcome handling.** Going from "return a float" to "return a normalized distribution over N labels the model has never seen before" exposed every brittle assumption in the parsing layer. Most of the bugs lived there, not in the prompt.
- **Calibration vs hedging tradeoff.** Clipping bounds the downside but also caps the upside on the events you're genuinely confident about. `[0.03, 0.97]` was picked from backtest, not vibes.
- **Solo time pressure.** No teammate to bounce prompts off, no second pair of eyes on the parser. Discipline came from the Brier loop — if a change didn't improve the score on the resolved set, it didn't ship.

## Accomplishments that we're proud of

- Multi-outcome native from the first commit, not bolted on.
- Full path from `git clone` to a live Render URL in under 17 hours, solo.
- Backtest harness that made every change measurable instead of vibes-based.
- The parser hasn't returned a malformed response yet across the resolved dataset.
- Brier 0.168 on the public `sample-resolved` set — inside the market consensus band (~0.15-0.22) on questions where the LLM cutoff is well past the resolution date, thanks to `:online` search grounding.

## What we learned

- Brier rewards humility surprisingly hard. The math on `(p - outcome)^2` means a 0.95 that turns out wrong costs you ~10x what a 0.7 that turns out wrong does. Most "improvements" to a forecasting agent are actually just learning to stop overclaiming.
- LLMs are decent at picking a reference class when you ask them to, and noticeably worse when you don't.
- A tight, local evaluation loop is worth more than any individual prompt trick.

## What's next for the project

- **Web search integration** for events whose resolution depends on facts the model's training cutoff doesn't cover.
- **Ensembling** across multiple models via OpenRouter and aggregating with a calibrated mean.
- **Category-specific routing** — sports, politics, and macro events probably want different prompts and possibly different models.
- **RAG over historical Kalshi data** to give the model an actual empirical base rate for similar past markets instead of asking it to guess one.

## Built with

Python 3.14, FastAPI, Uvicorn, Pydantic, OpenRouter, Claude Sonnet 4, Render.

## Try it yourself

```bash
git clone https://github.com/chanjoongx/prophet-hacks
cd prophet-hacks
pip install -r requirements.txt
cp .env.example .env   # add your OPENROUTER_API_KEY
uvicorn main:app --reload
```

Then point the Prophet CLI at `http://localhost:8000/predict`, or hit the deployed Render URL directly. Run `python scripts/backtest.py` to replay the resolved sample set and see Brier.

Source: https://github.com/chanjoongx/prophet-hacks
