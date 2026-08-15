#!/usr/bin/env bash
# Everything CI would run on Linux, if Linux had CI.
#
# It does not, and that is not an oversight to be fixed with a workflow file: the
# unit suite links strelka_display, which only exists when the editor is built,
# which on Linux pulls in the OptiX backend -- so a hosted runner cannot run any
# of this without the OptiX SDK headers, and NVIDIA does not permit
# redistributing them. Rendering needs a GPU on top of that.
#
# The cost of having no such gate is on record: the OptiX device code did not
# compile for months over a single undeclared identifier, and nothing said so.
# Until a self-hosted runner exists, this script is the gate -- run it before
# pushing anything that touches src/render/optix, src/shaders/optix, or a header
# shared with them.
#
#   tools/ci/linux_check.sh              # build + unit tests + smokes
#   tools/ci/linux_check.sh --ladder     # and the full parity ladder
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

: "${OPTIX_DIR:=$HOME/work/OptiX-SDK-9.1.0}"
export OPTIX_DIR
[ -d "$HOME/conan-venv/bin" ] && export PATH="$HOME/conan-venv/bin:$PATH"

ASSETS="${STRELKA_ASSETS:-$HOME/strelka_assets}"
BUILD="$ROOT/build/Release"
fail=0
step() { printf '\n=== %s ===\n' "$1"; }
check() { if [ "$1" -eq 0 ]; then echo "  ok"; else echo "  FAILED"; fail=1; fi; }

step "build"
# ninja has returned 0 here while a target failed, so the error count is the
# authority, not the exit status.
cmake --build "$BUILD" -j "$(nproc)" > /tmp/linux_check_build.log 2>&1
errors=$(grep -c 'error:' /tmp/linux_check_build.log)
echo "  compiler errors: $errors"
[ "$errors" -eq 0 ] && [ -x "$BUILD/StrelkaCLI" ]
check $?

step "optix device code produced"
for ir in OptixRender OptixRender_closest_hit; do
    test -s "$BUILD/optix/strelka_shaders_generated_${ir}.cu.optixir" || fail=1
done
check $fail

step "unit tests"
"$BUILD/unit_tests" > /tmp/linux_check_tests.log 2>&1
tail -3 /tmp/linux_check_tests.log | sed 's/^/  /'
grep -q "Status: SUCCESS" /tmp/linux_check_tests.log
check $?

step "cornell smoke"
# Exposure is stated rather than defaulted: the default ISO 100 / f4 / 1/100s
# through the ACES curve underflows this scene to exactly zero in 8-bit, so a
# smoke that only checked the file existed would pass on a black image -- which
# is what the macOS gate does today.
"$BUILD/StrelkaCLI" "$ASSETS/validation/cornell_box/cornell_box.glb" \
    -o /tmp/linux_check_cornell.exr -w 128 --height 96 --spp 16 \
    --tonemap none --exposure-iso 100 > /tmp/linux_check_cornell.log 2>&1
python3 - <<'PY'
import sys
sys.path.insert(0, "tools/feature_tests")
from exr_io import load_exr
a = load_exr("/tmp/linux_check_cornell.exr")
print("  mean %.5f  max %.4f  lit %.1f%%" % (a.mean(), a.max(), 100 * (a.sum(-1) > 1e-6).mean()))
sys.exit(0 if a.mean() > 1e-4 else 1)
PY
check $?

step "brainstem skinning smoke"
# animation_time 0.5 is load-bearing: at the clip start the skeleton is never
# dirtied, so this passes with skinning broken.
"$BUILD/StrelkaCLI" "$ASSETS/validation/brainstem/BrainStem.glb" \
    -o /tmp/linux_check_brainstem.exr -w 128 --height 96 --spp 8 \
    --animation-time 0.5 --tonemap none > /tmp/linux_check_brainstem.log 2>&1
grep -Eq 'Skinned geometry extent: [1-9]' /tmp/linux_check_brainstem.log
check $?

if [ "${1:-}" = "--ladder" ]; then
    step "parity ladder"
    python3 tools/parity/run_ladder.py --cli "$BUILD/StrelkaCLI" --suffix ci --spp 128
    check $?
fi

printf '\n%s\n' "$([ $fail -eq 0 ] && echo 'linux_check: OK' || echo 'linux_check: FAILED')"
exit $fail
