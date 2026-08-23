#!/bin/bash
# Bootstrap Contract (Principle 6): init.sh is done only when all four pass:
#   1. Can start   2. Can test   3. Can see progress   4. Can pick up next steps
set -e

cd "$(dirname "$0")/consistency-forecasting"

echo "=== [1/4] Installing dependencies ==="
pip install -r requirements.txt --break-system-packages
# If requirements.txt pins conflict with the environment, fall back to:
#   pip install -r requirements_loose.txt --break-system-packages
cp -n .env.example .env 2>/dev/null || true
echo "NOTE: fill in API keys in consistency-forecasting/.env before running any"
echo "      evaluation that calls a real model (OPENAI_API_KEY / ANTHROPIC_API_KEY /"
echo "      OPENROUTER_API_KEY etc.). Tests below do not require live keys."

echo "=== [2/4] Running tests (at least one must pass) ==="
pytest tests/test_hybrid_acd_forecaster.py -v

echo "=== [3/4] Verifying build / type-check ==="
# This is a Python project: no compile/build step. Use ruff for lint/type-adjacent checks.
ruff check src/forecasters/hybrid_acd_forecaster.py

echo "=== [4/4] Confirming harness state files exist ==="
cd ..
for f in AGENTS.md feature_list.json claude-progress.md; do
  if [ ! -f "$f" ]; then
    echo "MISSING: $f — harness is incomplete."
    exit 1
  fi
done

echo "=== Environment healthy. Read claude-progress.md and feature_list.json next. ==="
