#!/usr/bin/env bash
set -euo pipefail

echo "== sensitive keyword scan =="
rg -n \
  "company-internal-keyword|private-project-name|private/path|AKIA[0-9A-Z]{16}|sk-[A-Za-z0-9_-]{20,}|password\\s*=|api[_-]?key\\s*=|secret\\s*=|access[_-]?token\\s*=|auth[_-]?token\\s*=" \
  -S . \
  --glob '!data/raw_input/**' \
  --glob '!outputs/**' \
  --glob '!models/**' \
  --glob '!PretrainedModels/**' \
  --glob '!.venv/**' \
  --glob '!.uv-cache/**' \
  --glob '!scripts/__pycache__/**' \
  --glob '!scripts/audit_open_source.sh' || true

echo
echo "== large files outside ignored heavy directories =="
find . \
  -path './.git' -prune -o \
  -path './.venv' -prune -o \
  -path './.uv-cache' -prune -o \
  -path './outputs' -prune -o \
  -path './data/raw_input' -prune -o \
  -path './models' -prune -o \
  -path './PretrainedModels' -prune -o \
  -type f -size +10M -print

echo
echo "== git status =="
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  git status --short --ignored
else
  echo "not a git repository yet"
fi

echo
echo "audit complete"
