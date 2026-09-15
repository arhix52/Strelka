#!/usr/bin/env bash
# Windows-friendly export: uses imgui/1.91.4-docking when 1.92.8 is absent from conan-center.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WORKDIR="${TMPDIR:-/tmp}/strelka-conan-export-$$"
mkdir -p "$WORKDIR"
cleanup() { rm -rf "$WORKDIR"; }
trap cleanup EXIT

have_recipe() {
    conan cache path "$1" >/dev/null 2>&1
}

if have_recipe "imgui/1.92.9b-docking"; then
    echo "imgui/1.92.9b-docking already in cache"
else
    echo "Exporting imgui/1.92.9b-docking ..."
    conan download imgui/1.91.4-docking -r conancenter --only-recipe
    recipe_dir="$(conan cache path imgui/1.91.4-docking)"
    imgui_dir="${WORKDIR}/imgui_recipe"
    mkdir -p "${imgui_dir}"
    cp "${recipe_dir}/conanfile.py" "${recipe_dir}/conandata.yml" "${imgui_dir}/"
    if [[ -f "${recipe_dir}/CMakeLists.txt" ]]; then
        cp "${recipe_dir}/CMakeLists.txt" "${imgui_dir}/"
    else
        curl -fsSL -o "${imgui_dir}/CMakeLists.txt" \
            https://raw.githubusercontent.com/conan-io/conan-center-index/master/recipes/imgui/all/CMakeLists.txt
    fi
    IMGUI_CONANDATA="${imgui_dir}/conandata.yml" python - <<'PY'
import os
from pathlib import Path
path = Path(os.environ["IMGUI_CONANDATA"])
text = path.read_text()
entry = """  1.92.9b-docking:
    sha256: 90ded916bd57db2e0e171b6b098940a47c6f5042725dcdc67fb19940ca8bfdcc
    url: https://github.com/ocornut/imgui/archive/v1.92.9b-docking.tar.gz
"""
if "1.92.9b-docking:" not in text:
    text = text.replace("sources:\n", "sources:\n" + entry, 1)
    path.write_text(text)
PY
    (cd "${imgui_dir}" && conan export . --version=1.92.9b-docking)
fi

if have_recipe "glfw/3.5.1"; then
    echo "glfw/3.5.1 already in cache"
else
    echo "Exporting glfw/3.5.1 ..."
    conan download glfw/3.4 -r conancenter --only-recipe
    recipe_dir="$(conan cache path glfw/3.4)"
    glfw_dir="${WORKDIR}/glfw_recipe"
    mkdir -p "${glfw_dir}"
    cp "${recipe_dir}/conanfile.py" "${recipe_dir}/conandata.yml" "${glfw_dir}/"
    GLFW_CONANDATA="${glfw_dir}/conandata.yml" GLFW_CONANFILE="${glfw_dir}/conanfile.py" python - <<'PY'
import os
from pathlib import Path
path = Path(os.environ["GLFW_CONANDATA"])
conanfile = Path(os.environ["GLFW_CONANFILE"])
text = path.read_text()
entry = """  "3.5.1":
    url: "https://github.com/glfw/glfw/releases/download/3.5.1/glfw-3.5.1.zip"
    sha256: "ea79bc5feffc254c87291980c2d0bce9acebb68c4983b79f961dcd2cb8a611a0"
"""
if '"3.5.1":' not in text:
    text = text.replace("sources:\n", "sources:\n" + entry, 1)
    path.write_text(text)
recipe = conanfile.read_text()
import re
recipe, count = re.subn(
    r"\n        # don't force static link to libgcc if MinGW\n"
    r"        replace_in_file\(self, os\.path\.join\(self\.source_folder, \"src\", \"CMakeLists\.txt\"\),\n"
    r"                        \"target_link_libraries\(glfw PRIVATE \\\"-static-libgcc\\\"\)\", \"\"\)\n",
    "\n",
    recipe,
    count=1,
)
if count:
    conanfile.write_text(recipe)
PY
    (cd "${glfw_dir}" && conan export . --version=3.5.1)
fi

if have_recipe "imguizmo/cci.20260729"; then
    echo "imguizmo/cci.20260729 already in cache"
else
    echo "Exporting imguizmo/cci.20260729 ..."
    conan download imguizmo/cci.20231114 -r conancenter --only-recipe
    recipe_dir="$(conan cache path imguizmo/cci.20231114)"
    gizmo_dir="${WORKDIR}/imguizmo_recipe"
    mkdir -p "${gizmo_dir}"
    cp "${recipe_dir}/conanfile.py" "${recipe_dir}/conandata.yml" "${gizmo_dir}/"
    if [[ -f "${recipe_dir}/CMakeLists.txt" ]]; then
        cp "${recipe_dir}/CMakeLists.txt" "${gizmo_dir}/"
    else
        curl -fsSL -o "${gizmo_dir}/CMakeLists.txt" \
            https://raw.githubusercontent.com/conan-io/conan-center-index/master/recipes/imguizmo/all/CMakeLists.txt
    fi
    GIZMO_CMAKE="${gizmo_dir}/CMakeLists.txt" GIZMO_CONANDATA="${gizmo_dir}/conandata.yml" GIZMO_CONANFILE="${gizmo_dir}/conanfile.py" python - <<'PY'
import os
import re
from pathlib import Path
cmake = Path(os.environ["GIZMO_CMAKE"])
cdata = Path(os.environ["GIZMO_CONANDATA"])
conanfile = Path(os.environ["GIZMO_CONANFILE"])
cm = cmake.read_text()
if "set(SOURCE_DIR src/src)" not in cm:
    if "set(SOURCE_DIR src)" in cm:
        cm = cm.replace("set(SOURCE_DIR src)", "set(SOURCE_DIR src/src)", 1)
    else:
        cm = "set(SOURCE_DIR src/src)\n" + cm
if re.search(r"set\(CMAKE_CXX_STANDARD\s+\d+\)", cm):
    cm = re.sub(r"set\(CMAKE_CXX_STANDARD\s+\d+\)", "set(CMAKE_CXX_STANDARD 20)", cm)
elif "CMAKE_CXX_STANDARD" not in cm:
    cm = "set(CMAKE_CXX_STANDARD 20)\n" + cm
cmake.write_text(cm)
cf = conanfile.read_text()
needle = 'tc = CMakeToolchain(self)'
if needle in cf and "CMAKE_CXX_STANDARD" not in cf:
    cf = cf.replace(needle, needle + '\n        tc.variables["CMAKE_CXX_STANDARD"] = "20"', 1)
    conanfile.write_text(cf)
text = cdata.read_text()
entry = """  cci.20260729:
    sha256: cb59df243ba49c4183454d8f9fafb15ffb1a1c74a1a1e718f0b852f1345b9354
    url: https://github.com/CedricGuillemet/ImGuizmo/archive/5ab7676402ace03cdf930b2d972f59c7d03c6fa8.zip
"""
if "cci.20260729:" not in text:
    text = text.replace("sources:\n", "sources:\n" + entry, 1)
    cdata.write_text(text)
PY
    (cd "${gizmo_dir}" && conan export . --version=cci.20260729)
fi

echo "Local Conan packages ready."
