#!/bin/bash
#
# Give StrelkaEditor a name and an icon in the desktop's own menus.
#
# The window already identifies itself: GlfwDisplay::init() sets the Wayland
# app_id and the X11 WM_CLASS to "Strelka". What a compositor does with that is
# look for Strelka.desktop -- and until one exists it has a string and no icon,
# which is the "unknown" entry in the dock. This installs the pair.
#
# Per user, into ~/.local/share, because that needs no root and is what the XDG
# spec searches first. Re-running it is safe and is how an icon change or a moved
# build directory is picked up.
#
# Wayland has no other way: glfwSetWindowIcon() is documented as unsupported
# there, so the .desktop file is not a nicety, it is the mechanism.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SOURCE_ICON="${ROOT}/resources/icons/Strelka.png"
BINARY="${1:-${ROOT}/build/Release/StrelkaEditor}"

if [ ! -f "$SOURCE_ICON" ]; then
    echo "No icon at $SOURCE_ICON" >&2
    exit 1
fi
if [ ! -x "$BINARY" ]; then
    echo "No editor at $BINARY -- build first, or pass the path as the first argument." >&2
    exit 1
fi

ICON_ROOT="${XDG_DATA_HOME:-$HOME/.local/share}/icons/hicolor"
APPS_DIR="${XDG_DATA_HOME:-$HOME/.local/share}/applications"
mkdir -p "$APPS_DIR"

# hicolor is indexed by exact pixel size, so a 1254x1254 source in a directory
# named 256x256 is a theme that lies and a renderer that rescales badly. Write
# the sizes the spec lists and let the desktop pick.
for size in 512 256 128 64 48 32; do
    dir="${ICON_ROOT}/${size}x${size}/apps"
    mkdir -p "$dir"
    if command -v magick >/dev/null 2>&1; then
        magick "$SOURCE_ICON" -resize "${size}x${size}" "${dir}/Strelka.png"
    else
        python3 - "$SOURCE_ICON" "${dir}/Strelka.png" "$size" <<'PY'
import sys
from PIL import Image
source, destination, size = sys.argv[1], sys.argv[2], int(sys.argv[3])
Image.open(source).convert("RGBA").resize((size, size), Image.LANCZOS).save(destination)
PY
    fi
done

# StartupWMClass is the line that ties the two together: it is what the
# compositor matches the window's app_id against, and without it the running
# editor stays a second, nameless entry beside its own launcher.
cat > "${APPS_DIR}/Strelka.desktop" <<DESKTOP
[Desktop Entry]
Type=Application
Name=Strelka
GenericName=Path Tracer
Comment=GPU path tracer
Exec=${BINARY} %f
Icon=Strelka
Terminal=false
Categories=Graphics;3DGraphics;
MimeType=model/gltf+json;model/gltf-binary;
StartupWMClass=Strelka
DESKTOP

# Both are caches: without a refresh the file is on disk and the menu does not
# know. Neither is required to exist.
command -v update-desktop-database >/dev/null 2>&1 && update-desktop-database "$APPS_DIR" || true
command -v gtk-update-icon-cache >/dev/null 2>&1 && gtk-update-icon-cache -f -t "$ICON_ROOT" >/dev/null 2>&1 || true

echo "Installed ${APPS_DIR}/Strelka.desktop -> ${BINARY}"
echo "Installed icons under ${ICON_ROOT}/<size>/apps/Strelka.png"
