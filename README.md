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

Each `market` matches the exact label from `event.outcomes`. Values clipped to
`[0.03, 0.97]` to avoid Brier blow-ups on overconfident misses. Probabilities
do not need to sum to 1 — the eval server normalizes before scoring.

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

## License

MIT — see [`LICENSE`](LICENSE).
