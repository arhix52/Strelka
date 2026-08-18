# Vulkan HDR display

Linux and Windows use the Vulkan display backend. macOS continues to use Metal.
The editor selects HDR10 automatically when the window surface exposes
`A2B10G10R10_UNORM_PACK32` with `HDR10_ST2084_EXT`; otherwise it recreates the
swapchain in SDR using `B8G8R8A8_SRGB`.

The OptiX image remains scene-linear. CUDA tone maps directly into one of two
external RGBA16F Vulkan images, synchronized with exported timeline semaphores.
ImGui and the viewport are blended in a linear RGBA16F composition image. The
final Vulkan pass either converts Rec.709 to Rec.2020 and encodes ST2084/PQ, or
writes linear color to the SDR sRGB swapchain.

Display output controls are under Render Settings:

- `Auto`, `HDR10`, or `SDR`
- UI paper white and display peak luminance
- VRR diagnostics and the compositor-reported refresh range

Vulkan presentation remains FIFO. On Wayland, Mutter controls VRR; Strelka only
queries and reports its state and never changes monitor configuration.

## Validation

Build and run the tests with:

```shell
./build.sh Debug
cmake --build build/Debug --target StrelkaEditor unit_tests vulkan_cuda_interop_smoke
ctest --test-dir build/Debug --output-on-failure
```

The GPU smoke test verifies CUDA/Vulkan UUID matching, external RGBA16F memory,
timeline ordering, CUDA surface writes, Vulkan readback, and extent recreation.
It skips with CTest code 77 when the required CUDA/Vulkan interop is unavailable.

Linux runtime validation is performed on NVIDIA/Wayland. Windows code paths are
compiled conditionally, but HDR and external-handle runtime validation requires
a Windows NVIDIA system.
