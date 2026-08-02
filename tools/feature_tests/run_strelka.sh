#!/usr/bin/env bash
# Render every feature-test scene with StrelkaCLI.
#
#   tools/feature_tests/run_strelka.sh [scenes/feature_tests] [name-filter]
#
# StrelkaCLI must run from build/Release: it resolves ./metal/shaders/*.metallib
# relative to the working directory. The .toml files carry absolute paths for
# the scene and the output, so only the binary's own cwd matters.

set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# The binary runs from build/Release, so a relative scene root would resolve
# against the wrong directory once we cd there.
OUT_ROOT="$(cd "${1:-$REPO/scenes/feature_tests}" && pwd)"
FILTER="${2:-}"
CLI_DIR="$REPO/build/Release"
CLI="$CLI_DIR/StrelkaCLI"

if [[ ! -x "$CLI" ]]; then
    echo "StrelkaCLI not found at $CLI"
    echo "It lives on the bdpt_dev branch; cherry-pick 15f2d28 and build."
    exit 1
fi

pass=0; fail=0
for toml in "$OUT_ROOT"/*/*.toml; do
    [[ -e "$toml" ]] || { echo "No .toml under $OUT_ROOT — run build_features.py first."; exit 1; }
    name="$(basename "$toml" .toml)"
    [[ -n "$FILTER" && "$name" != *"$FILTER"* ]] && continue

    printf '%-24s ' "$name"
    log="$(dirname "$toml")/${name}_strelka.log"
    if (cd "$CLI_DIR" && "$CLI" --config "$toml") >"$log" 2>&1; then
        echo "ok"
        pass=$((pass+1))
    else
        echo "FAILED (see $log)"
        fail=$((fail+1))
    fi
done

echo
echo "$pass ok, $fail failed"
[[ $fail -eq 0 ]]
