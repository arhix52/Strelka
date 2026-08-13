#!/usr/bin/env bash
# Run clang-tidy on src/ and tests/ (never third_party/ or external/).
#
#   tools/run_clang_tidy.sh              # whole tree
#   tools/run_clang_tidy.sh file1.cpp …  # explicit files
#
# Requires compile_commands.json at the repo root (./build.sh Release symlinks it).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ ! -f compile_commands.json ]]; then
    echo "run_clang_tidy: compile_commands.json not found — run ./build.sh Release first." >&2
    exit 1
fi

CLANG_TIDY="${CLANG_TIDY:-clang-tidy}"
if ! command -v "$CLANG_TIDY" >/dev/null 2>&1; then
    echo "run_clang_tidy: clang-tidy not found (brew install llvm)" >&2
    exit 1
fi

HEADER_FILTER="${STRELKA_TIDY_HEADER_FILTER:-(src|tests)/.*}"
JOBS="${STRELKA_TIDY_JOBS:-$(sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 4)}"

is_project_source() {
    local path="$1"
    case "$path" in
        src/*|tests/*) ;;
        *) return 1 ;;
    esac
    case "$path" in
        */third_party/*|*/external/*) return 1 ;;
    esac
    case "$path" in
        *.cpp|*.mm|*.h|*.hpp) ;;
        *) return 1 ;;
    esac
    return 0
}

run_one() {
    local file="$1"
    "$CLANG_TIDY" -p "$ROOT" \
        --config-file="$ROOT/.clang-tidy" \
        --header-filter="$HEADER_FILTER" \
        "$file"
}

if [[ "$#" -gt 0 ]]; then
    status=0
    while [[ "$#" -gt 0 ]]; do
        rel="${1#"$ROOT"/}"
        shift
        if ! is_project_source "$rel"; then
            continue
        fi
        if ! run_one "$rel"; then
            status=1
        fi
    done
    exit "$status"
fi

RUN_CLANG_TIDY="$(command -v run-clang-tidy 2>/dev/null || true)"
if [[ -n "$RUN_CLANG_TIDY" ]]; then
    exec "$RUN_CLANG_TIDY" -p "$ROOT" \
        -config-file="$ROOT/.clang-tidy" \
        -header-filter="$HEADER_FILTER" \
        -j "$JOBS" \
        "$ROOT/src" "$ROOT/tests"
fi

status=0
mapfile -d '' files < <(find src tests \( -name '*.cpp' -o -name '*.mm' \) -print0)
for f in "${files[@]}"; do
    if ! run_one "$f"; then
        status=1
    fi
done
exit "$status"
