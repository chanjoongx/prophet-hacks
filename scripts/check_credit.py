"""Quick credit check for the OpenRouter API key in .env.

Usage:
    python scripts/check_credit.py
"""

from __future__ import annotations

import io
import os
import sys

import httpx
from dotenv import load_dotenv

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

load_dotenv()
key = os.environ.get("OPENROUTER_API_KEY")
if not key:
    print("ERROR: OPENROUTER_API_KEY not set in .env")
    sys.exit(1)

r = httpx.get(
    "https://openrouter.ai/api/v1/auth/key",
    headers={"Authorization": f"Bearer {key}"},
    timeout=10,
)
if r.status_code != 200:
    print(f"ERROR: HTTP {r.status_code}: {r.text}")
    sys.exit(1)

data = r.json()["data"]
limit = data["limit"]
usage = data["usage"]
remaining = data["limit_remaining"]
pct_used = (usage / limit * 100) if limit else 0

# At ensemble rate ~$0.066/call, how many calls remain
calls_remaining = remaining / 0.066 if remaining else 0
calls_solo = remaining / 0.022 if remaining else 0

print(f"OpenRouter key: {data['label']}")
print(f"  Limit:     ${limit:.2f}")
print(f"  Used:      ${usage:.2f}  ({pct_used:.1f}%)")
print(f"  Remaining: ${remaining:.2f}")
print(f"  Daily:     ${data.get('usage_daily', 0):.2f}")
print(f"  Weekly:    ${data.get('usage_weekly', 0):.2f}")
print()
print(f"Estimated calls remaining:")
print(f"  Ensemble (3 models): {calls_remaining:.0f} calls")
print(f"  Solo Sonnet:         {calls_solo:.0f} calls")
print()
if remaining < 5:
    print("WARNING: low balance. Consider USE_ENSEMBLE=0 on Render.")
elif remaining < 15:
    print("Note: ~25% remaining; safe but worth monitoring.")
else:
    print("Plenty of budget for the remaining eval.")
