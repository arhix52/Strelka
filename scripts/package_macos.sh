#!/usr/bin/env bash
# Install the macOS app and CLI into dist/Strelka and zip them for distribution.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="${STRELKA_BUILD_DIR:-$ROOT/build/Release}"
PREFIX="${STRELKA_PACKAGE_PREFIX:-$ROOT/dist/Strelka}"
ARCH="$(uname -m)"
ZIP="${STRELKA_PACKAGE_ZIP:-$ROOT/dist/Strelka-macos-${ARCH}.zip}"

if [[ ! -f "${BUILD}/StrelkaCLI" ]]; then
    echo "error: ${BUILD}/StrelkaCLI missing — run ./build.sh Release first" >&2
    exit 1
fi
if [[ ! -d "${BUILD}/metal/shaders" ]]; then
    echo "error: ${BUILD}/metal/shaders missing — Metal shaders were not built" >&2
    exit 1
fi

rm -rf "${PREFIX}"
mkdir -p "$(dirname "${PREFIX}")"

cmake --install "${BUILD}" --prefix "${PREFIX}"

APP="${PREFIX}/Strelka.app"
if [[ ! -x "${APP}/Contents/MacOS/Strelka" ]]; then
    echo "error: ${APP} was not installed" >&2
    exit 1
fi
if [[ "$(/usr/libexec/PlistBuddy -c 'Print :CFBundleIdentifier' "${APP}/Contents/Info.plist")" == "" ]]; then
    echo "error: bundle identifier missing" >&2
    exit 1
fi
for resource in \
    "metal/shaders/wavefront.metallib" \
    "metal/shaders/tonemapper.metallib" \
    "metal/shaders/skinning.metallib" \
    "metal/shaders/fullScreen.metal" \
    "materialx/libraries/stdlib/stdlib_defs.mtlx" \
    "Strelka.icns" \
    "default_layout.ini"; do
    if [[ ! -f "${APP}/Contents/Resources/${resource}" ]]; then
        echo "error: bundle resource missing: ${resource}" >&2
        exit 1
    fi
done

# cmake --install may not refresh LICENSE/README if the root install rules ran
# against a stale tree; ensure they are present.
cp -f "${ROOT}/LICENSE" "${ROOT}/README.md" "${PREFIX}/"

cd "$(dirname "${PREFIX}")"
rm -f "${ZIP}"
zip -qry "${ZIP}" "$(basename "${PREFIX}")"

echo "Packaged ${ZIP}"

# Smoke from the install tree (same criteria as CI: exit 0 + non-empty PNG).
SCENE="${ROOT}/scenes/validation/cornell_box/cornell_box.glb"
if [[ -f "${SCENE}" ]]; then
    SMOKE_OUT="${TMPDIR:-/tmp}/strelka_package_smoke.png"
    "${PREFIX}/StrelkaCLI" "${SCENE}" -o "${SMOKE_OUT}" -w 64 --height 48 --spp 4
    test -s "${SMOKE_OUT}"
    echo "Package smoke OK -> ${SMOKE_OUT}"
else
    echo "warning: ${SCENE} missing — skipped package smoke" >&2
fi
