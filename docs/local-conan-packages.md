# Locally exported Conan packages

Two dependencies are pinned to versions conan-center does not publish. Prefer:

```bash
./scripts/export_local_conan.sh
```

`./build.sh` calls that script before `conan install`. Re-run the script only if
the Conan cache is wiped. Drop both pins once conan-center publishes
1.92.9-docking (or newer) and an ImGuizmo newer than 2023.

## Why these pins

### imgui/1.92.9b-docking

conan-center stops at 1.92.8-docking, which predates `imgui_impl_metal4.*` —
the Metal 4 render backend. Without it the UI pass cannot leave Metal 3.

### imguizmo/cci.20260729

Forced by the ImGui bump: the 2023 conan-center package calls
`ImGui::BeginChildFrame` and the old `ImDrawList::AddPolyline` signature, both
removed in 1.92. Upstream ImGuizmo has kept up (5ab7676402, 2026-07-29).

## Manual export (Conan 2)

Only needed if the script cannot run (no network, offline recipe edits, …).

### imgui

```sh
conan download imgui/1.92.8-docking -r conancenter --only-recipe
recipe="$(conan cache path imgui/1.92.8-docking)"
mkdir imgui_recipe && cd imgui_recipe
cp "$recipe"/{conanfile.py,conandata.yml} .
# Ensure CMakeLists.txt is present (copy from the recipe dir or CCI).
# Add under sources: in conandata.yml:
#   1.92.9b-docking:
#     core:
#       sha256: 90ded916bd57db2e0e171b6b098940a47c6f5042725dcdc67fb19940ca8bfdcc
#       url: https://github.com/ocornut/imgui/archive/v1.92.9b-docking.tar.gz
conan export . --version=1.92.9b-docking
```

### imguizmo

```sh
conan download imguizmo/cci.20231114 -r conancenter --only-recipe
recipe="$(conan cache path imguizmo/cci.20231114)"
mkdir imguizmo_recipe && cd imguizmo_recipe
cp "$recipe"/{conanfile.py,conandata.yml} .
# Edit CMakeLists.txt:
#   set(SOURCE_DIR src/src)   # sources live under src/ after strip_root
#   set(CMAKE_CXX_STANDARD 17)
# Add under sources: in conandata.yml:
#   cci.20260729:
#     sha256: cb59df243ba49c4183454d8f9fafb15ffb1a1c74a1a1e718f0b852f1345b9354
#     url: https://github.com/CedricGuillemet/ImGuizmo/archive/5ab7676402ace03cdf930b2d972f59c7d03c6fa8.zip
conan export . --version=cci.20260729
```
