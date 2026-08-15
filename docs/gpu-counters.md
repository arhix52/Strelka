# Metal GPU hardware counters

Use a Release build with Metal API Validation and Shader Validation disabled.
Hardware-counter collection is device-wide, so keep the desktop quiet and take
an idle control before every renderer capture.

## Command-line capture

`tools/gpu_counters.sh` creates a minimal Instruments template, records the
`Performance Limiters` counter set, and prints duration-weighted averages. It
waits for `STRELKA_RENDER_BEGIN` in `--run` mode so scene loading, acceleration
structure construction, and the first sample are outside the measurement.

```bash
# Device-wide control.
tools/gpu_counters.sh --idle 2 > /tmp/gpu-idle.txt

# kids_room; quote the complete renderer command as one argument.
tools/gpu_counters.sh --run \
  'build/Release/StrelkaCLI -c /private/tmp/strelka_perf.toml -s scenes/kids_room/kids_room.gltf -o /private/tmp/kids-counters.exr' \
  2 > /tmp/gpu-kids.txt

# Keep a trace for Instruments instead of deleting it after the summary.
STRELKA_KEEP_TRACE=/private/tmp/kids-counters.trace \
  tools/gpu_counters.sh --run \
  'build/Release/StrelkaCLI -c /private/tmp/strelka_perf.toml -s scenes/kids_room/kids_room.gltf -o /private/tmp/kids-counters.exr' \
  2

# Keep the native trace and skip the slow XML export and text summary.
STRELKA_KEEP_TRACE=/private/tmp/kids-counters.trace \
STRELKA_SKIP_EXPORT=1 \
  tools/gpu_counters.sh --run \
  'build/Release/StrelkaCLI -c /private/tmp/strelka_perf.toml -s scenes/kids_room/kids_room.gltf -o /private/tmp/kids-counters.exr' \
  2
```

Compare with the idle averages, but don't blindly subtract them. The counters
belong to the GPU, not the attached PID; WindowServer and other Metal processes
otherwise contaminate the values. Occupancy, limiter, utilization, cache-rate,
and other percentage counters are ratios and aren't additive. An idle control
shows whether that background load is material; only additive quantities such
as bandwidth can sometimes be baseline-adjusted, and only when the control is
stable. Keep scene, camera, resolution, sample count, power source, and thermal
state fixed between comparisons.

The default profile is `13` on Xcode 26 and newer and `3` on older Xcode. Xcode
26 uses profile 13 for the M4-compatible `Performance Limiters` preset; the
legacy profile 3 produces `Selected counter profile is not supported on target
device` on M4. The script treats that warning and an empty counter table as
errors instead of printing an empty report.

Two overrides are available when another Xcode or GPU requires them:

```bash
STRELKA_GPU_COUNTER_PROFILE=13 tools/gpu_counters.sh --idle 2
STRELKA_GPU_COUNTER_TEMPLATE=/absolute/path/Strelka.tracetemplate \
  tools/gpu_counters.sh --idle 2
```

## Instruments UI

Apple's templates do not enable performance counters by default. In Instruments:

1. Open `Game Performance` (or `Metal System Trace`) and select the target Mac
   and process.
2. Click and hold Record, then choose **Recording Options**.
3. Under the Metal/GPU options, set **Counter Set** to
   **Performance Limiters**. This also records the utilization counters.
4. Leave **Shader Timeline** disabled for a counter-only measurement. Enable it
   only when shader correlation is needed, and compare like with like.
5. Record the steady-state render interval. Use **File > Save As Template** if
   the configuration should be passed to `STRELKA_GPU_COUNTER_TEMPLATE`.

This is the setting Apple documents in
[Analyzing the performance of your Metal app](https://developer.apple.com/documentation/xcode/analyzing-the-performance-of-your-metal-app/).

To verify a saved trace from the command line, its table of contents must show
both the selected set and a nonempty hardware-counter schema:

```bash
xcrun xctrace export --input /private/tmp/kids-counters.trace --toc \
  --output /private/tmp/kids-counters-toc.xml
rg 'Counter Set: Performance Limiters|counter-profile=' \
  /private/tmp/kids-counters-toc.xml
```

On the M4 Pro/Xcode 26.6 test machine, the expected metadata is
`counter-profile="13"`, `shader-profiler="0"`; the set exposes limiter,
utilization, occupancy, cache, bandwidth, MMU, and ray-tracing activity
counters. A trace that merely lists the `metal-gpu-counter-intervals` schema is
not sufficient—the schema can exist with zero rows when the profile is wrong.
