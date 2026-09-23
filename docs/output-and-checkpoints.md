# CLI output and checkpoints

For a final still, use `.exr` output and `--postprocess-package` (or
`[output].postprocess_package = true`). The beauty EXR contains float32
scene-linear Strelka RGB, with opaque alpha; no exposure, tone curve or display
gamma is baked into it. Chromaticities are not tagged: glTF sRGB inputs are
decoded to linear RGB, but externally authored linear textures are not all
normalised to a single OCIO working space. Assign/convert a working space in
the compositor according to the assets. `<stem>.preview.png` is the
display-referred view made
with the configured exposure, tonemapper and gamma. `<stem>.render.json` records
the scene path, image dimensions, actual spp, sampler, filter, depth and camera.
The package is explicit so benchmarking and validation runs do not spend time
encoding an extra PNG. Spatial upscaling and animation/audit sequences are
currently refused for a package; the former's output path is display-referred,
and the latter has no single-camera/spp result to describe.

For a long, ordinary path-traced still:

```bash
build/Release/StrelkaCLI scene.glb -o final.exr --spp 4096 \
    --spp-per-launch 4 --checkpoint-spp 256 --postprocess-package

# After an interruption, using the same scene, camera and render settings:
build/Release/StrelkaCLI scene.glb -o final.exr --spp 4096 \
    --spp-per-launch 4 --checkpoint-spp 256 \
    --resume final.checkpoint.stc --postprocess-package
```

The EXR checkpoint is for inspection. The `.stc` file is the resume source:
it holds the linear accumulated mean, completed sample count, a settings/asset
signature, and a checksum. It is atomically replaced at each checkpoint. Resume
restores the GPU accumulation buffer and sample index before the next launch;
the target `--spp` is a **total**, not additional spp. Sample batch size and
target total may change. A different camera, resolution, filter, depth, sampler,
scene/sidecar/texture timestamp or size is rejected before tracing. This is a
same-machine working checkpoint, not an archive format or a cryptographic hash
of every external asset; if external geometry or texture contents change while
preserving file timestamps and sizes, start a fresh render.

Resume is limited to a static, full-resolution PT still without MetalFX,
ReSTIR, SHaRC, debug views, audit or capture. Those modes have temporal state
outside the accumulated beauty buffer, so restoring only the image would not
reproduce their result. Checkpoint files are float32 RGBA (16 bytes per pixel)
plus a small header; budget disk space accordingly (about 1 GiB at 8K UHD).
