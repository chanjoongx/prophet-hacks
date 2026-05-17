# Reproducible container for the Prophet Hacks 2026 forecasting agent.
#
# Build:
#   docker build -t chanjoongx-prophet-hacks .
#
# Run (port 8000):
#   docker run --rm -p 8000:8000 \
#     -e OPENROUTER_API_KEY=sk-or-v1-... \
#     -e FORECAST_MODEL=anthropic/claude-sonnet-4:online \
#     chanjoongx-prophet-hacks
#
# Then POST to http://localhost:8000/predict per the contract in README.md.

FROM python:3.12-slim

WORKDIR /app

# System deps for lxml etc. (some ai-prophet transitive deps need them).
RUN apt-get update \
 && apt-get install -y --no-install-recommends gcc libxml2-dev libxslt1-dev \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

COPY main.py ./
COPY scripts ./scripts

EXPOSE 8000

# Container reads PORT env var (Render convention) or defaults to 8000.
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
