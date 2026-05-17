# chanjoongx — Prophet Hacks 2026 Forecasting Agent

A calibrated multi-outcome forecasting agent for the
[Prophet Hacks 2026](https://www.prophethacks.com/) Forecasting Track.

The agent exposes `POST /predict` accepting the Event schema produced by
`prophet forecast retrieve` and returns a calibrated probability per outcome,
graded against Brier.

Live endpoint: <https://chanjoongx-prophet-hacks.onrender.com/predict>

## Architecture

```
POST /predict (Event JSON)
        ↓
┌───────────────────────────────────────────────────────────┐
│ Tier 1 — OpenRouter :online ensemble (parallel, weight 1) │
│   • anthropic/claude-sonnet-4:online  ×2  (self-consistency)
│   • openai/gpt-4o:online                                  │
│   • google/gemini-2.5-flash:online                        │
│   ⇒ each call uses max_results=10 Exa search context      │
└───────────────────────────────────────────────────────────┘
        │ all failed (rare) ↓
┌───────────────────────────────────────────────────────────┐
│ Tier 2 — Anthropic direct (claude-sonnet-4-20250514)      │
│   independent infra, no web search                        │
└───────────────────────────────────────────────────────────┘
        │ also failed ↓
┌───────────────────────────────────────────────────────────┐
│ Tier 3 — OpenAI direct (gpt-4o)                           │
│   independent infra, no web search                        │
└───────────────────────────────────────────────────────────┘
        │ also failed ↓
   uniform 1/N — endpoint NEVER returns 5xx
```

Across all paths the response goes through the same parser:

- **Hybrid residual-mass coercion** — outcomes the LLM did not explicitly
  mention get `max(true_residual_mass, 0.5/N)`. Keeps total ~1.0 so server
  normalization is a no-op without amplifying confidently-wrong calls.
- **Asymmetric N-aware clipping** — ceiling `0.95` universal; floor `0.05`
  for binary, `0` for multi-outcome.
- **Balanced-brace JSON extractor** — tolerates the markdown citations the
  `:online` models append after their JSON payload.
- **NaN / Inf / percent / whitespace defenses** — every path returns a
  Pydantic-validated `Prediction`.

## Endpoint contract

`POST /predict` — body is a single Event JSON object.

Response:

```json
{
  "probabilities": [
    {"market": "Cleveland", "probability": 0.62},
    {"market": "Detroit",   "probability": 0.38}
  ],
  "rationale": "weighted ensemble: 3/4 search + 0 direct ...",
  "p_yes": 0.62
}
```

- Every `market` value exactly matches one of the event's `outcomes` labels.
- Every `probability` is a decimal in `[0, 1]`.
- Total may not sum exactly to 1 (the Prophet eval server normalizes per the
  Quick Start docs); our hybrid residual-mass logic keeps it close.
- The extra `p_yes` field is a back-compat extra so the `ai-prophet 0.1.5`
  CLI — which reads `float(result["p_yes"])` — also works.

## Quickstart (local dev)

```bash
python -m venv .venv
.venv/Scripts/activate            # Windows
# source .venv/bin/activate       # macOS/Linux

pip install -r requirements.txt
cp .env.example .env              # then put your OpenRouter key in .env

# Pull a sample slate (no credentials needed)
prophet forecast retrieve --dataset sample-sports -o events.json

# Serve the agent
uvicorn main:app --host 0.0.0.0 --port 8000

# In another shell, drive the agent through the official CLI
prophet forecast predict --events events.json \
    --agent-url http://localhost:8000/predict
```

## Local Brier backtest (the actual dev tool)

```bash
prophet forecast retrieve --dataset sample-resolved --include-resolved -o resolved.json
python scripts/build_actuals.py resolved.json actuals.json

# Score the live local server end-to-end (replays 26 resolved events,
# parallel, prints overall + per-category Brier).
python scripts/backtest.py
```

Or score the deployed Render endpoint directly:

```bash
python scripts/production_backtest.py https://chanjoongx-prophet-hacks.onrender.com
```

## One-command organizer eval

Per the Prophet Hacks submission rules ("Include a script the organizers can
run to evaluate your agent in a standard environment"):

```bash
OPENROUTER_API_KEY=sk-or-v1-... ./evaluate.sh
```

The script creates a clean venv, installs deps, pulls `sample-sports`, boots
the agent, and runs the official `prophet forecast predict` against it.

Or via Docker:

```bash
docker build -t chanjoongx-prophet-hacks .
docker run --rm -p 8000:8000 \
  -e OPENROUTER_API_KEY=sk-or-v1-... \
  chanjoongx-prophet-hacks
```

## Configuration

Set in `.env` for local dev, or in the Render dashboard for the deployed service.

| Var | Required | Default | Notes |
|---|---|---|---|
| `OPENROUTER_API_KEY` | **yes** | — | Primary key for the Tier-1 ensemble. The hackathon-issued $50 credit key works. |
| `ANTHROPIC_API_KEY` | no | — | Tier-2 fallback only (active path is disabled by default — see `USE_DIRECT_PROVIDERS`). |
| `OPENAI_FALLBACK_API_KEY` | no | — | Tier-3 fallback. Distinct slot from `OPENAI_API_KEY` (which is the OpenRouter alias). |
| `FORECAST_MODEL` | no | `anthropic/claude-sonnet-4` | Used only when `USE_ENSEMBLE=0` (single-model mode). |
| `ENSEMBLE_MODEL_1..4` | no | sonnet:online ×2, gpt-4o:online, gemini-2.5-flash:online | Override individual ensemble members. |
| `USE_ENSEMBLE` | no | `1` | Set to `0` to disable the ensemble and use `FORECAST_MODEL` only. |
| `USE_DIRECT_PROVIDERS` | no | `0` | Set to `1` to put Anthropic/OpenAI direct in the *active* ensemble at weight `DIRECT_PROVIDER_WEIGHT`. Disabled by default because backtest showed they dilute confident-correct calls. |
| `DIRECT_PROVIDER_WEIGHT` | no | `0.4` | Weight for the direct-provider samples when active. |
| `WEB_MAX_RESULTS` | no | `10` | OpenRouter `:online` plugin `max_results`. |
| `PER_MODEL_TIMEOUT_S` | no | `20` | Per-model request timeout (seconds). |
| `ANTHROPIC_FALLBACK_MODEL` | no | `claude-sonnet-4-20250514` | Anthropic direct model id. |
| `OPENAI_FALLBACK_MODEL` | no | `gpt-4o` | OpenAI direct model id. |

## Deployment

Render free tier works out of the box; the `render.yaml` blueprint declares
the service name, region, build / start commands, and the env var slots.

```bash
# After connecting the repo to Render as a Blueprint:
# 1. Render auto-builds: pip install -r requirements.txt
# 2. Start: uvicorn main:app --host 0.0.0.0 --port $PORT
# 3. Paste OPENROUTER_API_KEY (and optionally ANTHROPIC_API_KEY,
#    OPENAI_FALLBACK_API_KEY) in the dashboard.
```

A 5-minute UptimeRobot ping on `/health` is recommended to keep the free-tier
container warm between eval requests.

## AI tools disclosure

Per the Prophet Hacks 2026 rules (Section 7: *"AI tools (coding assistants,
LLM agents, etc.) are allowed and encouraged for building"*), this project
was built solo by [@chanjoongx](https://github.com/chanjoongx) using
[Claude Code](https://claude.com/claude-code) as a coding assistant during
the 32-hour build window. All architectural decisions, prompt engineering,
calibration choices, and the Brier evaluation methodology are the author's;
Claude Code was used the way one would use an autocomplete or pair-programming
tool.

The project's runtime LLMs (the models that actually produce forecasts) are
configurable via `.env` / Render env vars. The shipped defaults are a 4-way
:online ensemble across three providers, none of which is Claude Code.

## License

MIT — see [`LICENSE`](LICENSE).
