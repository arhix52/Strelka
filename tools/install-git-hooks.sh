#!/usr/bin/env bash
# Point this repo at .githooks/ (pre-commit -> tools/precommit.sh).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

chmod +x .githooks/pre-commit tools/precommit.sh tools/run_clang_tidy.sh

git config core.hooksPath .githooks

echo "Git hooks installed (core.hooksPath=.githooks)."
echo "Pre-commit runs clang-tidy on staged src/*.cpp, src/*.mm, tests/*.cpp."
echo "Opt out: git commit --no-verify"
