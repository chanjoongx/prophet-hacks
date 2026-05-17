#!/usr/bin/env bash
# evaluate.sh — bring up the agent in a clean environment and verify it answers.
#
# Per the Prophet Hacks 2026 submission rules:
#   "Include a script the organizers can run to evaluate your agent in a
#    standard environment."
#
# Usage:
#   OPENROUTER_API_KEY=sk-or-v1-... ./evaluate.sh
#
# What it does:
#   1. Creates a fresh venv, installs deps from requirements.txt
#   2. Pulls a small dataset (sample-sports, 16 events) via the official CLI
#   3. Boots the agent on http://localhost:8000
#   4. Runs `prophet forecast predict --agent-url ...` against the agent
#   5. Prints success / failure summary
#
# Exits non-zero on any failure. Cleans up the local server on exit.

set -euo pipefail

if [ -z "${OPENROUTER_API_KEY:-}" ] && [ -z "${OPENAI_API_KEY:-}" ]; then
  echo "ERROR: set OPENROUTER_API_KEY (preferred) or OPENAI_API_KEY before running" >&2
  exit 1
fi

cleanup() {
  if [ -n "${SERVER_PID:-}" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "==> Stopping local server (pid $SERVER_PID)"
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

echo "==> Setting up isolated venv at .venv-eval/"
python -m venv .venv-eval
# shellcheck disable=SC1091
source .venv-eval/bin/activate 2>/dev/null || source .venv-eval/Scripts/activate

echo "==> Installing dependencies"
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt

echo "==> Pulling sample-sports events (16 events, no credentials needed)"
prophet forecast retrieve --dataset sample-sports -o events.json

echo "==> Booting agent on http://localhost:8000"
uvicorn main:app --host 127.0.0.1 --port 8000 > evaluate-server.log 2>&1 &
SERVER_PID=$!

echo "==> Waiting for server to become ready..."
for i in {1..20}; do
  if curl -sf http://localhost:8000/health > /dev/null 2>&1; then
    echo "    server is up"
    break
  fi
  sleep 1
  if [ "$i" = 20 ]; then
    echo "ERROR: server never became ready" >&2
    cat evaluate-server.log >&2
    exit 1
  fi
done

echo "==> Running prophet forecast predict against the local agent"
prophet forecast predict \
  --events events.json \
  --agent-url http://localhost:8000/predict \
  -o predictions.json \
  --timeout 60

echo ""
echo "==> Predictions written to predictions.json"
echo "==> First prediction:"
python -c "import json; print(json.dumps(json.load(open('predictions.json'))['predictions'][0], indent=2))"

echo ""
echo "==> evaluate.sh completed successfully"
