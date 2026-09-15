#!/bin/bash
set -euo pipefail

if [ "$#" -ne 1 ] && [ "$#" -ne 2 ]; then
    echo "Usage: $0 <Debug|Release|Profile> [clean]"
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

case "$build_type" in
    Debug)
        conan_build_type="Debug"
        metal_profile_sources="OFF"
        ;;
    Release)
        conan_build_type="Release"
        metal_profile_sources="OFF"
        ;;
    Profile)
        # Keep profiling metadata out of build/Release, which is the only tree
        # package_macos.sh accepts for distribution.
        conan_build_type="Release"
        metal_profile_sources="ON"
        ;;
    *)
        echo "error: build type must be Debug, Release, or Profile" >&2
        exit 1
        ;;
esac

git submodule update --init --recursive

# Pins that conan-center does not publish yet (GLFW, ImGui Metal 4, ImGuizmo).
./scripts/export_local_conan.sh

conan install . -c tools.cmake.cmaketoolchain:generator=Ninja \
    -c tools.system.package_manager:mode=install \
    -c tools.system.package_manager:sudo=True \
    --build=missing --settings=build_type="$conan_build_type"

build_dir="$ROOT/build/$build_type"
generator_dir="$ROOT/build/$conan_build_type/generators"
mkdir -p "$build_dir"
cd "$build_dir"

if [ "$clean_option" == "clean" ]; then
    cmake --build . --target clean
fi

# shellcheck disable=SC1091
source "$generator_dir/conanbuild.sh"

cmake ../.. -G Ninja \
    -DCMAKE_TOOLCHAIN_FILE="$generator_dir/conan_toolchain.cmake" \
    -DCMAKE_BUILD_TYPE="$conan_build_type" \
    -DSTRELKA_METAL_PROFILE_SOURCES="$metal_profile_sources"

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
