# Strelka

GPU path tracer for macOS (Metal). Scenes are glTF/GLB; analytic lights and
environment live in a `<stem>_light.json` sidecar, curves in `<stem>_curves.bin`.

Two apps ship from a default build:

| Binary | Role |
|--------|------|
| `Strelka.app` | Interactive ImGui editor |
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
./build.sh Release   # source-free distribution build
./build.sh Debug
./build.sh Profile   # optimized build with private Metal profiling symbols
```

The macOS build targets Apple silicon and macOS 26 because the renderer requires
Metal 4. Release identity can be overridden with `STRELKA_MARKETING_VERSION`,
`STRELKA_BUILD_NUMBER`, and `STRELKA_BUNDLE_IDENTIFIER` CMake cache variables.

Binaries and runtime assets land in `build/Release/` (or `build/Debug/`):

```text
build/Release/
  Strelka.app/
  StrelkaCLI
  unit_tests
  metal/shaders/*.metallib
  default_layout.ini
```

`Profile` uses Release optimization but writes to `build/Profile/`. The app's
metallibs contain line information while shader sources and symbols are kept in
`build/Profile/metal/profiling/*.metallib.dSYM`; that directory is never
installed. Import those companions in Xcode when correlating a capture.

Launch from that directory (or from anywhere — assets resolve relative to the
executable):

```bash
build/Release/Strelka.app/Contents/MacOS/Strelka -s scenes/validation/cornell_box/cornell_box.glb
build/Release/StrelkaCLI scenes/validation/cornell_box/cornell_box.glb -o out.exr -w 512 --height 384 --spp 256
SPDLOG_LEVEL=debug build/Release/StrelkaCLI ...
```

`StrelkaCLI` accepts a TOML config (`-c` / `--config`); every flag overrides the
matching key. `--checkpoint-spp N` writes both `<stem>.checkpoint.<ext>` and a
resumable `<stem>.checkpoint.stc`. Continue with `--resume <stem>.checkpoint.stc
--spp TOTAL`; `TOTAL` includes samples already saved. For a postproduction
deliverable, `--postprocess-package` writes a scene-linear beauty EXR, a
tonemapped PNG preview and `<stem>.render.json` with render/camera metadata.
See [output and checkpoint details](docs/output-and-checkpoints.md).

## Package (macOS)

After a Release build:

```bash
./scripts/package_macos.sh
# -> dist/Strelka-macos-<arch>.zip
```

Packaging refuses non-Release and `Profile` builds. It audits the Strelka shader
directories for embedded source, debug/reflection sections, local source-tree
paths, and stray `.metal`, `.air`, `.metallibsym`, `.dSYM`, or `.gputrace`
artifacts. MaterialX's upstream source implementations remain runtime data under
`materialx/libraries/` and are outside this Strelka-shader check.

Layout inside the zip (prefix root — same as the build tree for asset paths):

```text
Strelka/
  Strelka.app/
  StrelkaCLI
  metal/shaders/*.metallib
  materialx/libraries/
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

Generate a repeatable mixed animated/idle crowd from the bundled BrainStem asset:

```bash
python3 tools/make_animation_crowd.py --count 64 --animated-ratio 0.75 \
  --duration-scale 0.8 1.2 -o build/profiles/brainstem-crowd-64.glb
build/Release/Strelka.app/Contents/MacOS/Strelka \
  -s build/profiles/brainstem-crowd-64.glb
```

The generator shares the source GLB payload, duplicates each character's node/skin
graph, gives animated characters independent clips in several duration buckets,
and writes an overview camera and matching light sidecar. Use **Animations → Play**
to exercise continuous skinning and BLAS updates; idle characters remain in bind pose.
For a headless median over 120 independently phased frames:

```bash
build/Release/StrelkaCLI build/profiles/brainstem-crowd-64.glb \
  -o /tmp/crowd.exr -w 1280 --height 720 --depth 1 --animation-frames 120
```

For a large game-style crowd, quantize distant characters into a fixed number of
shared poses. Each bucket skins and refits one BLAS; its characters remain separate
TLAS instances and may still move independently:

```bash
python3 tools/make_animation_crowd.py --count 1024 --animated-ratio 0.9 \
  --shared-pose-buckets 16 -o build/profiles/brainstem-crowd-1024.glb
build/Release/StrelkaCLI build/profiles/brainstem-crowd-1024.glb \
  -o /tmp/crowd.exr -w 1280 --height 720 --depth 1 --animation-frames 120
```

Omit `--shared-pose-buckets` for the worst case where every animated character
owns an independent pose, skinned vertex stream, and refittable BLAS.

## License

MIT — see [LICENSE](LICENSE). Third-party trees keep their own licenses.
