#!/usr/bin/env bash
# Capture one steady-state Metal frame per scene for Instruments / xctrace.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CLI="${ROOT}/build/Release/StrelkaCLI"
OUT="${HOME}/strelka_metal_profile/traces"
mkdir -p "$OUT"

run_scene() {
  local name="$1"
  local scene="$2"
  local camera="${3:-0}"
  local trace="${OUT}/${name}_sobol_metal.trace"

  echo "== ${name} -> ${trace}"
  xctrace record \
    --template 'Metal System Trace' \
    --output "$trace" \
    --launch -- "$CLI" "$scene" \
      -o "/dev/null" \
      -w 1280 --height 720 \
      --spp 12 --depth 4 \
      --camera "$camera" \
      --sampler sobol \
      --tonemap none \
    || true
}

run_scene iso_bathroom "${ROOT}/scenes/iso_bathroom/iso_bathroom.gltf" 1
run_scene kids_room "${ROOT}/scenes/kids_room/kids_room.gltf" 0
if [[ -f "${HOME}/pine_scene/polyhaven_pine_fir_forest.gltf" ]]; then
  run_scene pine_scene "${HOME}/pine_scene/polyhaven_pine_fir_forest.gltf" 0
fi

echo "Traces in ${OUT}"
