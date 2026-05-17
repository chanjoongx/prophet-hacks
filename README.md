# chanjoongx — Prophet Hacks 2026 Forecasting Agent

A calibrated multi-outcome forecasting agent built for the
[Prophet Hacks 2026](https://www.prophethacks.com/) Forecasting Track.

The agent exposes `POST /predict` accepting the Event schema produced by
`prophet forecast retrieve` and returns a probability per outcome,
scored against Brier.

## Quickstart

```bash
python -m venv .venv
.venv/Scripts/activate            # Windows
# source .venv/bin/activate       # macOS/Linux

pip install -r requirements.txt
cp .env.example .env              # then put your OpenRouter key in .env

# Pull a sample slate
prophet forecast retrieve --dataset sample-sports -o events.json

# Serve the agent
uvicorn main:app --host 0.0.0.0 --port 8000

# In another shell — hit the agent through the official CLI
prophet forecast predict --events events.json \
    --agent-url http://localhost:8000/predict
```

## Local Brier backtest

```bash
prophet forecast retrieve --dataset sample-resolved --include-resolved -o resolved.json
python scripts/build_actuals.py resolved.json actuals.json

prophet forecast predict --events resolved.json \
    --agent-url http://localhost:8000/predict \
    -o resolved_predictions.json

prophet forecast evaluate --submission resolved_predictions.json --actuals actuals.json
```

## Endpoint contract

`POST /predict` — body is a single `Event` JSON.

Response:

```json
{
  "probabilities": [
    {"market": "Cleveland", "probability": 0.62},
    {"market": "Detroit",   "probability": 0.38}
  ],
  "rationale": "..."
}
```

Each `market` matches the exact label from `event.outcomes`. Probabilities are
clipped with an asymmetric, N-aware bound: ceiling `0.95` universally; floor
`0.05` for binary events, `0` for multi-outcome — applying a floor to every
wrong outcome in a 30-way race destroys mass on the actual outcome after
server-side normalization. Probabilities do not need to sum to 1; they are
scored as-is per the official spec. An additional `p_yes` field is returned
alongside `probabilities` for back-compat with the `ai-prophet 0.1.5` CLI.

## Configuration

Set in `.env`:

| Var | Required | Default | Notes |
|-----|----------|---------|-------|
| `OPENROUTER_API_KEY` | yes (or `OPENAI_API_KEY`) | — | Hackathon $50 credit key |
| `FORECAST_MODEL` | no | `anthropic/claude-sonnet-4` | Any OpenRouter model id |
| `OPENROUTER_BASE_URL` | no | `https://openrouter.ai/api/v1` | Override for direct OpenAI |

## Deployment

The free tier on [Render](https://render.com) works out of the box.
Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`

## AI tools disclosure

Per the Prophet Hacks 2026 rules (Section 7: "AI tools (coding assistants,
LLM agents, etc.) are allowed and encouraged for building"), this project
was built solo by [@chanjoongx](https://github.com/chanjoongx) using
[Claude Code](https://claude.com/claude-code) as a coding assistant during
the 32-hour build window. All architectural decisions, prompt engineering,
calibration choices, and the local Brier evaluation methodology are the
author's; Claude Code was used for code scaffolding, refactoring, and
debugging — the same way one would use an autocomplete or pair-programming
tool.

The project's runtime LLM (the model that actually produces forecasts) is
configurable via `.env`; defaults to `anthropic/claude-sonnet-4:online` via
OpenRouter.

## License

MIT — see [`LICENSE`](LICENSE).
