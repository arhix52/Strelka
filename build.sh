#!/bin/bash
set -euo pipefail

if [ "$#" -ne 1 ] && [ "$#" -ne 2 ]; then
    echo "Usage: $0 <build_type> [clean]"
    exit 1
fi

build_type="$1"
clean_option="${2:-}"

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

# Function to convert the input parameter to start with a capital letter
ucfirst() {
    echo "$1" | awk '{print toupper(substr($0,1,1)) tolower(substr($0,2))}'
}

build_type=$(ucfirst "$build_type")

git submodule update --init --recursive

# Pins that conan-center does not publish yet (GLFW, ImGui Metal 4, ImGuizmo).
./scripts/export_local_conan.sh

conan install . -c tools.cmake.cmaketoolchain:generator=Ninja \
    -c tools.system.package_manager:mode=install \
    -c tools.system.package_manager:sudo=True \
    --build=missing --settings=build_type="$build_type"

cd build/"$build_type"

if [ "$clean_option" == "clean" ]; then
    cmake --build . --target clean
fi

# shellcheck disable=SC1091
source ./generators/conanbuild.sh

cmake ../.. -G Ninja \
    -DCMAKE_TOOLCHAIN_FILE=generators/conan_toolchain.cmake \
    -DCMAKE_BUILD_TYPE="$build_type"

cmake --build .

# clangd / Cursor: prefer a workspace-root compile_commands.json when present.
# Debug is the default IDE configuration; Release builds leave an existing
# Debug symlink alone so IntelliSense stays pointed at a TU graph that matches
# the F5 launch config.
if [ -f compile_commands.json ]; then
    if [ "$build_type" = "Debug" ] || [ ! -e "$ROOT/compile_commands.json" ]; then
        ln -sfn "build/${build_type}/compile_commands.json" "$ROOT/compile_commands.json"
    fi
fi

echo "Build completed (${build_type})."
