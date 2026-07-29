#!/usr/bin/env bash
#
# Batch-render all BDPT test scenes with PT, BDPT, and VCM integrators.
# Must be run from build/Release/ directory.
#
# Usage:
#   cd build/Release
#   bash ../../scripts/render_bdpt_tests.sh [--spp N] [--scene NAME]
#
# Options:
#   --spp N        Override samples per pixel (default: use per-scene value)
#   --scene NAME   Only render a specific scene (e.g. "cornell_box")
#   --integrator X Only use a specific integrator (pt, bdpt, vcm)
#   --width W      Override render width
#   --height H     Override render height

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
SCENES_DIR="$REPO_ROOT/scenes/bdpt_tests"
CLI="./StrelkaCLI"

# Defaults
SPP_OVERRIDE=""
SCENE_FILTER=""
INTEGRATOR_FILTER=""
WIDTH_OVERRIDE=""
HEIGHT_OVERRIDE=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --spp)       SPP_OVERRIDE="$2"; shift 2 ;;
        --scene)     SCENE_FILTER="$2"; shift 2 ;;
        --integrator) INTEGRATOR_FILTER="$2"; shift 2 ;;
        --width)     WIDTH_OVERRIDE="$2"; shift 2 ;;
        --height)    HEIGHT_OVERRIDE="$2"; shift 2 ;;
        *)           echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Check we're in the right directory
if [[ ! -x "$CLI" ]]; then
    echo "Error: $CLI not found. Run this script from build/Release/"
    echo "  cd build/Release && bash ../../scripts/render_bdpt_tests.sh"
    exit 1
fi

# Create output directory
mkdir -p output

SCENES=(cornell_box glass_caustics metal_sphere mixed_materials small_light indirect_cove caustic_pool)
INTEGRATORS=(pt bdpt vcm)

# Apply filters
if [[ -n "$SCENE_FILTER" ]]; then
    SCENES=("$SCENE_FILTER")
fi
if [[ -n "$INTEGRATOR_FILTER" ]]; then
    INTEGRATORS=("$INTEGRATOR_FILTER")
fi

TOTAL=0
FAILED=0

echo "================================================================"
echo "  BDPT Test Suite"
echo "================================================================"
echo ""

for scene in "${SCENES[@]}"; do
    for integrator in "${INTEGRATORS[@]}"; do
        TOML="$SCENES_DIR/$scene/${scene}_${integrator}.toml"

        if [[ ! -f "$TOML" ]]; then
            echo "SKIP: $TOML not found"
            continue
        fi

        OUTPUT="output/${scene}_${integrator}.exr"

        echo "--- Rendering: $scene / $integrator ---"

        # Build CLI arguments
        ARGS=(-c "$TOML")

        if [[ -n "$SPP_OVERRIDE" ]]; then
            ARGS+=(--spp "$SPP_OVERRIDE")
        fi
        if [[ -n "$WIDTH_OVERRIDE" ]]; then
            ARGS+=(-w "$WIDTH_OVERRIDE")
        fi
        if [[ -n "$HEIGHT_OVERRIDE" ]]; then
            ARGS+=(--height "$HEIGHT_OVERRIDE")
        fi

        TOTAL=$((TOTAL + 1))

        if "$CLI" "${ARGS[@]}"; then
            echo "  OK: $OUTPUT"
        else
            echo "  FAILED: $scene / $integrator"
            FAILED=$((FAILED + 1))
        fi

        echo ""
    done
done

echo "================================================================"
echo "  Results: $((TOTAL - FAILED))/$TOTAL passed"
if [[ $FAILED -gt 0 ]]; then
    echo "  $FAILED FAILED"
fi
echo "  Output: $(pwd)/output/"
echo "================================================================"
echo ""
echo "Compare results visually or with an image diff tool:"
echo "  For each scene, PT/BDPT/VCM should converge to the same image."
echo "  BDPT should show caustics that PT misses (especially glass_caustics, small_light)."
