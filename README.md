# Strelka

GPU path tracer for macOS (Metal). Scenes are glTF/GLB; analytic lights and
environment live in a `<stem>_light.json` sidecar, curves in `<stem>_curves.bin`.

Two apps ship from a default build:

| Binary | Role |
|--------|------|
| `StrelkaEditor` | Interactive ImGui editor |
| `StrelkaCLI` | Headless renderer (EXR/PNG, TOML config) |

Windows/Linux still contain an OptiX/CUDA backend; it is **not** the packaged
product path and is not covered by the macOS CI job.

## Prerequisites (macOS)

- Xcode with the Metal toolchain (`xcrun -sdk macosx --find metal`)
- [Conan 2](https://conan.io/), [Ninja](https://ninja-build.org/)
- `git submodule update --init --recursive` (or use `./build.sh`, which does it)

Two Conan packages are pinned to versions conan-center does not publish yet
(`imgui/1.92.9b-docking`, `imguizmo/cci.20260729`). `./build.sh` exports them
when missing; details are in [docs/local-conan-packages.md](docs/local-conan-packages.md).

## Build

```bash
./build.sh Release   # or: ./build.sh Debug
```

Binaries and runtime assets land in `build/Release/` (or `build/Debug/`):

```text
build/Release/
  StrelkaEditor
  StrelkaCLI
  unit_tests
  metal/shaders/*.metallib
  default_layout.ini
```

Launch from that directory (or from anywhere — assets resolve relative to the
executable):

```bash
cd build/Release
./StrelkaEditor -s ../../scenes/validation/cornell_box/cornell_box.glb
./StrelkaCLI ../../scenes/validation/cornell_box/cornell_box.glb -o out.exr -w 512 --height 384 --spp 256
SPDLOG_LEVEL=debug ./StrelkaCLI ...
```

`StrelkaCLI` accepts a TOML config (`-c` / `--config`); every flag overrides the
matching key. See `RenderConfig` in `src/cli/HeadlessApp.h`.

## Package (macOS)

After a Release build:

```bash
./scripts/package_macos.sh
# -> dist/Strelka-macos-<arch>.zip
```

Layout inside the zip (prefix root — same as the build tree for asset paths):

```text
Strelka/
  StrelkaEditor
  StrelkaCLI
  metal/shaders/*.metallib
  default_layout.ini
  LICENSE
  README.md
```

## IDE setup (VS Code / Cursor)

1. Install the **clangd** extension (Microsoft C/C++ IntelliSense is disabled in
   [`.vscode/settings.json`](.vscode/settings.json)).
2. Run the **Full Build (Debug)** task once (`Terminal → Run Task…`). That runs
   Conan, configures Ninja, builds, and symlinks `compile_commands.json` for clangd.
3. Press **F5** → **Launch StrelkaEditor (Debug)**. Leave the scene path empty to
   start with an empty document.

Useful tasks: **CMake Build (Debug/Release)**, **Run unit_tests**, **CLI smoke (Release)**.

## Tests

```bash
cd build/Release
ctest
# or:
./unit_tests
./unit_tests -tc="<test case name>"
```

Image-level feature tests vs Blender Cycles live under `tools/feature_tests/`
and are run manually — they are not part of CI (no golden images in the repo).

## GPU profiling

The working Xcode/Instruments counter preset, command-line capture workflow, and
idle-baseline rules are documented in [docs/gpu-counters.md](docs/gpu-counters.md).

## License

MIT — see [LICENSE](LICENSE). Third-party trees keep their own licenses.
