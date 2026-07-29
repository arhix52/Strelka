# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build Commands

**macOS (primary dev platform):**
```bash
# Full build (installs deps, configures, builds)
./build.sh Release    # or Debug

# Manual rebuild (after initial setup)
cd build/Release && source ./generators/conanbuild.sh
cmake --build .

# Build with clean
./build.sh Release clean
```

**Windows:**
```bash
conan install . --build=missing --settings=build_type=Debug
cd build
cmake .. -G "Visual Studio 17 2022" -DCMAKE_TOOLCHAIN_FILE=generators\conan_toolchain.cmake
cmake --build . --config Debug
```

**CMake options:** `STRELKA_BUILD_EDITOR` (ON), `STRELKA_BUILD_CLI` (ON), `STRELKA_BUILD_HYDRA` (OFF), `STRELKA_BUILD_TESTS` (ON)

**Run tests:**
```bash
cd build/Release && ctest
# or directly: ./unit_tests
```

**Run editor/CLI:** Must launch from `build/Release/` (Metal shader .metallib files are loaded via relative paths `./metal/shaders/*.metallib`).

**After git checkout of .metal files:** `touch src/shaders/metal/*.metal` before building to force metallib recompilation.

## Architecture

Strelka is a cross-platform path tracing renderer: OptiX (Win/Linux) + Metal (macOS), with an ImGui-based editor.

### Module Map
- `src/foundation/` — Logging (spdlog), `SettingsManager` (string key-value store, **non-const** `getAs<T>()`)
- `src/scene/` — Scene data: meshes, materials, lights, cameras
- `src/sceneloader/` — glTF loader (tinygltf)
- `src/material/` — Header-only cross-platform BSDF system (CUDA/Metal/CPU). Key: `bsdf.h` (4-function API: init/sample/eval/pdf), `material_math.h` (platform macros)
- `src/shaders/optix/` — CUDA/OptiX shaders (.cu)
- `src/shaders/metal/` — Metal shaders (.metal), compiled to .metallib
- `src/render/` — Render abstraction with `optix/` and `metal/` backends
- `src/display/` — Display abstraction with `opengl/` and `metal/` backends
- `src/editor/` — Interactive editor app with UI panels (Viewport, RenderSettings, Animation, Property)
- `src/cli/` — Headless CLI renderer (TOML config, cxxopts args)

### Rendering Algorithms
- **Path Tracing** (integrator type 0): Standard unidirectional
- **BDPT** (integrator type 1): Multi-kernel — camera subpath, light subpath, connect
- **VCM** (integrator type 2): BDPT + photon merging (spatial hash grid, Epanechnikov kernel)

### Data Flow
glTF → `GltfLoader` → `Scene` → `MetalRender`/`OptixRender` (GPU buffers, BLAS, TLAS) → ray trace → accumulate → `Display` → ImGui overlay

### Cross-Platform Material System
`src/material/include/strelka/material/material_math.h` has `#ifdef` sections for CUDA, Metal, and CPU. The `THREAD_REF` macro provides Metal `thread` address-space qualifiers. Metal uses `sqrt`/`cos`/`sin` (not `sqrtf`/`cosf`/`sinf`) — aliases are in the Metal section.

### Dependencies
Managed by Conan 2.x (`conanfile.py`). Key: glm, spdlog, tinygltf, imgui (docking), glfw, doctest. External: metal-cpp (manual download to `third_party/metal_cpp/`), ImGuiFileDialog (git submodule at `third_party/imgui_file_dialog/`).

## Code Conventions

- **C++17**, 4-space indent, 120 column limit, Allman braces
- **Naming**: CamelCase for classes/structs/namespaces, `m` prefix for private members (e.g., `mMeshes`)
- **clang-format** config at `.clang-format` (includes are NOT auto-sorted)
- **clang-tidy** config at `.clang-tidy` (warnings as errors)
- Pointers left-aligned: `int* ptr`

## Metal Shader Pitfalls

- Metal `float3` = 16 bytes in structs. CPU `Scene::Vertex` uses `glm::float3` = 12 bytes, stride 32. Use `device char*` with byte offsets for raw vertex buffers.
- `packed_float3` must be cast to `float3` before math operations.
- ALL reference params in Metal functions need explicit `thread` qualifier — use `THREAD_REF` macro.
- Never name a subdirectory `metal/` — macOS resolves `#include "metal/..."` to Metal.framework.
- Autoreleased command buffers: `retain()` if needed past autorelease pool drain, then `release()` after use.
- In `bdpt_common.h`: `using namespace metal;` must come before `#include <strelka/material/bsdf.h>`.

## Blender Export

```bash
blender --background file.blend --python scripts/blend2strelka.py -- --output DIR
```
Exports glTF + `_light.json` for lights + env HDRI.
