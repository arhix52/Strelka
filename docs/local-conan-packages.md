# Locally exported Conan packages

Two dependencies are pinned to versions conan-center does not publish. Both
recipes are the conan-center ones with an added version entry; recreate them with
the commands below if the Conan cache is wiped.

## imgui/1.92.9b-docking

conan-center stops at 1.92.8-docking, which predates `imgui_impl_metal4.*` — the
Metal 4 render backend, added upstream on 2026-07-07 and first tagged in
v1.92.9-docking. Without it the UI pass cannot leave Metal 3, because the older
backend takes `id<MTLCommandBuffer>` and `id<MTLRenderCommandEncoder>`.

```sh
mkdir imgui_recipe && cd imgui_recipe
cp "$(conan cache path --folder=export imgui/1.92.8-docking)"/{conanfile.py,conandata.yml} .
curl -sLO https://raw.githubusercontent.com/conan-io/conan-center-index/master/recipes/imgui/all/CMakeLists.txt
# prepend to conandata.yml under sources::
#   1.92.9b-docking:
#     core:
#       sha256: 90ded916bd57db2e0e171b6b098940a47c6f5042725dcdc67fb19940ca8bfdcc
#       url: https://github.com/ocornut/imgui/archive/v1.92.9b-docking.tar.gz
conan export . --version=1.92.9b-docking
```

## imguizmo/cci.20260729

Forced by the ImGui bump: the 2023 conan-center package calls `ImGui::BeginChildFrame`
and the old `ImDrawList::AddPolyline` signature, both removed in 1.92. Upstream
ImGuizmo has kept up, so this is a local export of its head (5ab7676402, 2026-07-29).

Two edits to the conan-center CMakeLists are needed:

- `set(SOURCE_DIR src/src)` — upstream moved its sources into a `src/` subfolder,
  which lands one level below the Conan source folder after `strip_root`.
- `set(CMAKE_CXX_STANDARD 17)` — the new `ImVectorEditor.cpp` needs more than C++11.

```sh
mkdir imguizmo_recipe && cd imguizmo_recipe
cp "$(conan cache path --folder=export imguizmo/cci.20231114)"/{conanfile.py,conandata.yml} .
curl -sLO https://raw.githubusercontent.com/conan-io/conan-center-index/master/recipes/imguizmo/all/CMakeLists.txt
# apply the two edits above, then add to conandata.yml under sources::
#   cci.20260729:
#     sha256: cb59df243ba49c4183454d8f9fafb15ffb1a1c74a1a1e718f0b852f1345b9354
#     url: https://github.com/CedricGuillemet/ImGuizmo/archive/5ab7676402ace03cdf930b2d972f59c7d03c6fa8.zip
conan export . --version=cci.20260729
```

Drop both once conan-center publishes 1.92.9-docking and an ImGuizmo newer than 2023.
