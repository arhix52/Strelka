#!/usr/bin/env bash
# Pre-commit checks for Strelka. Called from .githooks/pre-commit.
#
# Checks (staged files under src/ and tests/ only):
#   1. clang-tidy (our .clang-tidy; third_party/ and external/ are skipped)
#
# Opt out for one commit:
#   git commit --no-verify
#   SKIP_PRECOMMIT=1 git commit
#   STRELKA_SKIP_TIDY=1 git commit     # skip clang-tidy only
#
# Install: ./tools/install-git-hooks.sh
set -euo pipefail

if [[ "${SKIP_PRECOMMIT:-}" == "1" ]]; then
    exit 0
fi

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ ! -f compile_commands.json ]]; then
    echo "precommit: compile_commands.json missing — run ./build.sh Release once." >&2
    exit 1
fi

mapfile -t staged < <(git diff --cached --name-only --diff-filter=ACMRTUXB)

tidy_files=()
for path in "${staged[@]}"; do
    case "$path" in
        src/*.cpp|src/*.mm|tests/*.cpp) tidy_files+=("$path") ;;
    esac
done

if [[ "${#tidy_files[@]}" -eq 0 ]]; then
    echo "precommit: no staged src/*.cpp, src/*.mm, or tests/*.cpp — nothing to check."
    exit 0
fi

echo "precommit: clang-tidy on ${#tidy_files[@]} staged file(s)…"

if [[ "${STRELKA_SKIP_TIDY:-}" == "1" ]]; then
    echo "precommit: STRELKA_SKIP_TIDY=1 — skipping clang-tidy."
    exit 0
fi

if ! "$ROOT/tools/run_clang_tidy.sh" "${tidy_files[@]}"; then
    echo >&2
    echo "precommit: clang-tidy failed." >&2
    echo "  fix issues above, or: git commit --no-verify" >&2
    echo "  full tree: tools/run_clang_tidy.sh" >&2
    exit 1
fi

echo "precommit: OK"
