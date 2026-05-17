## Inspiration

Prediction markets like Kalshi are one of the few places where being just *slightly* better calibrated than the crowd has a quantitative, leaderboard-graded value. The Prophet Hacks Forecasting Track sharpens that into a tight loop: events come in, you return probabilities, Brier score grades you, and the loop runs continuously for two weeks. I wanted to see how far a single-person agent could push that loop in a weekend — not by being clever, but by exploiting every angle of *web-grounded* reasoning at once and being ruthless about what actually moved Brier in the local backtest.

## What it does

The agent exposes a single `POST /predict` endpoint that consumes the official Prophet `Event` schema (the same payload returned by `prophet forecast retrieve`) and responds with a calibrated probability per outcome.

It handles binary markets (Yes/No) and arbitrary multi-outcome markets (e.g., the 30-way NHL Calder Trophy race) on the same code path, with response shape guaranteed by Pydantic: every outcome label appears, every probability lives in `[0, 1]`, total is close to 1.0 so the eval server's normalization is effectively a no-op. The endpoint also returns a back-compat `p_yes` field so older `ai-prophet 0.1.5` CLI consumers that read `float(result["p_yes"])` keep working.

On the public `sample-resolved` dataset, the agent's production Brier ranges roughly **0.04 (excluding one search-noise outlier) to 0.09 (including it)** versus the snapshot market-consensus baseline around **0.15–0.22**.

## How we built it

The architecture is intentionally boring in structure and aggressive in inference. Most of the code is parsing and fallback paths; the interesting decisions are the ensemble shape and the calibration prompt.

**Tier 1 — OpenRouter `:online` ensemble (parallel, equal weight):** Claude Sonnet 4 × 2 (self-consistency), GPT-4o, and Gemini 2.5 Flash, each with the `:online` suffix and `max_results=10` so every call gets ~10 Exa-backed web results before reasoning. Independent search backends (Brave/Tavily, Bing, Google) are the real diversity here — when one model's first hit is misleading, the other models' different search engines provide the corrective signal.

**Tier 2/3 — independent-infrastructure fallback chain:** if every OpenRouter call fails, the agent walks Anthropic Sonnet direct → OpenAI GPT-4o direct → uniform `1/N`. Tested with mocked outages. (Active-mode direct providers were measured in backtest and disabled: no-search models don't know recent events and dilute the search-grounded correct answers more than they anchor wrong ones.)

**Hybrid residual-mass coercion:** for outcomes the LLM didn't explicitly mention in its JSON, the parser fills with `max(true_residual_mass, 0.5/N)` — a humility floor at half the uniform fallback. Empirically beats both the naive 1/N fill (which inflated sums and got deflated by server normalization) and pure residual mass (which amplified confidently-wrong calls into 1.7+ Brier disasters).

**Asymmetric N-aware clipping:** ceiling `0.95` universal, floor `0.05` binary / `0` multi-outcome. A floor on every wrong outcome in a 30-way race destroys the actual outcome's mass under normalization.

**Tetlock-style calibration prompt:** outside view → inside view → devil's advocate → date check → distribute mass. Anchored to specific probability values (5, 10, 20, 35, 50, 65, 80, 90, 95%) to break 0.5-clustering. Today's UTC date is injected into every user prompt because that was the highest-ROI single line change in the Halawi et al. forecasting paper.

**Defensive output parser:** balanced-brace JSON extractor that ignores the markdown citations `:online` responses tack on, case-insensitive + whitespace-tolerant outcome label matching, NaN/Inf filtering, automatic percent-to-decimal correction. Every error path returns a uniform-fallback `Prediction` so the endpoint never returns 5xx — missed events would tank the completion-rate factor in the leaderboard formula.

**Per-model 20s timeout** to defend the prophethacks.com submit-endpoint forecast-check (25s budget, NOT the 10-minute eval budget). Bounds tail latency even when one model gets slow.

**Local Brier backtest harness** replaying `sample-resolved` against the live server. Every prompt and clipping change earned its place through a measured Brier delta. Several "obviously good" ideas (uniform shrinkage, tighter binary ceiling, active-mode no-search ensemble members) lost in backtest and got reverted.

Deployed on Render free tier with auto-deploy on git push; `Dockerfile` and `evaluate.sh` included for organizer reproducibility per the submission rules.

## Challenges we ran into

**The schema docs contradict themselves.** Quick Start says "probabilities normalized before scoring", Custom Agent says "should sum to 1", participant guide says "OpenAI-compatible endpoint." An organizer clarified on Discord that "OpenAI-compatible" is loose phrasing — actual contract is the per-event POST with the explicit `{probabilities: [...]}` response. I built to that and added the `p_yes` field defensively to satisfy both the new spec and the older v0.1.5 CLI that reads `result["p_yes"]`.

**Ensemble doesn't fix systematic search misinformation.** When `:online` lands the LLM on a wrong source, additional samples of the *same* model trust the same wrong source. The genuine variance reduction came from *different search backends*, not more samples of one model — that insight reshaped the ensemble composition midway through the build.

**Calibration vs hedging is a real tradeoff, not a theoretical one.** Pure residual-mass coercion (sum stays ~1.0, server normalization is a no-op) won big on confident-correct cases but amplified the few confident-wrong cases into Brier 1.7+ disasters. The hybrid floor at `0.5/N` was the compromise — picked by backtest, not by feel.

**Multi-outcome handling exposed every brittle assumption in the parser.** Going from "return a float" to "return a normalized distribution over N labels the model may not have explicitly named in its output" is where the bugs lived. The residual-mass logic was rewritten three times before backtest agreed it was right.

**Solo time pressure.** No teammate to bounce prompts off, no second pair of eyes on the parser. Discipline came from the Brier loop — if a change didn't improve the score on `sample-resolved`, it didn't ship. Several theoretically clever ideas reverted because the data said so.

## Accomplishments that we're proud of

Multi-outcome native from the first commit, not bolted on. The 4-way `:online` ensemble with `max_results=10` and the hybrid residual-mass coercion is mathematically correct and empirically the best of three variants I measured against backtest. The 3-tier provider-redundant fallback chain (OpenRouter ensemble → Anthropic direct → OpenAI direct → uniform) has not returned a 5xx across any of the production backtests.

Every architectural decision is backed by a measured Brier delta — including the decisions to *not* ship features that seemed obviously useful but lost in the data (uniform shrinkage, no-search active ensemble members, tighter binary ceiling). The local backtest harness was the actual development tool, not the prompt itself.

Full path from `git clone` to a live Render URL with three-provider redundancy in under 24 hours, solo, with the entire empirical journey documented in commit messages.

## What we learned

Brier rewards humility surprisingly hard. The math on `(p - outcome)^2` means a 0.95 that turns out wrong costs you about 10× what a 0.7 that turns out wrong does. Most "improvements" to a forecasting agent are actually just learning to stop overclaiming.

Search backend diversity > model diversity. Different LLMs reading the same wrong source agree on the same wrong answer; different search engines surface different sources, and that's where the ensemble actually pays its bills.

The fastest way to lose Brier is to forget the date. Injecting today's UTC date into the user prompt was a one-line change that materially reduced "model confused training cutoff with event date" errors.

A tight, local evaluation loop is worth more than any individual prompt trick. Without `sample-resolved` to measure against, every change is a guess.

## What's next for chanjoongx Prophet Forecaster

**Category-specific calibration**, especially for "vote count" style events (SCOTUS, US Senate confirmations) where `:online` search consistently lands on the wrong tally and the LLMs go confidently wrong together. Constrained-output decoding with a domain prior is the obvious fix.

**Cross-provider Opus tier for hard events:** a Haiku-class classifier decides "is this event subtle enough to need Opus?" and routes accordingly. Cost-targeted, not blanket.

**Devil's-advocate two-stage** (the experimental `variant_e.py` in the repo) triggered only when the ensemble lands on `max(p) ≥ 0.85`. Cheap insurance against confident-wrong on rare categories; needs a backtest pass against a larger resolved set than 26 events before shipping.

**RAG over historical Kalshi data** to give the model an actual empirical base rate for similar past markets instead of asking it to guess one — the single largest gap between this agent and a real superforecaster ensemble.
