# Metal backend ownership

`MetalRender` orchestrates the frame (`prepare → deform → integrate → post →
present`). Domain classes own and mutate their GPU resources.

## Domains

- `MetalTextures`: material maps, mip/BC encoding, and disk cache.
- `MetalMaterials`: GPU material tables and material scene flags.
- `MetalGeometry`: vertex/index buffers, geometry entries, and mesh records.
- `MetalAccelStructure`: BLAS/TLAS, instance buffers, motion geometry, and
  scratch allocations.
- `MetalLights`: analytic light buffer.
- `MetalEnvironment`: dome texture, IBL alias table, and PDF scale.
- `MetalSkinning`: joint matrices, skinning pipeline, and skinned vertex writes.
- `MetalFrameUniforms`: camera, jitter, exposure, and SHARC uniforms.
- `MetalWavefrontIntegrator`: path queues, specialized pipelines, and encode.
- `MetalPostProcess`: tonemap, guide, display, and upscale textures.
- `MetalScenePreparation`: build-stage state only; no GPU resource ownership.

`Metal4Context`, `MetalFxContext`, and `MetalBuffer` are API boundary/adaptor
objects rather than scene domains.

## Contracts

- Geometry → Accel: descriptors name vertex-buffer offsets and counts.
  Skinning writes vertex contents; Accel owns refit/rebuild.
- Textures → Materials: Materials receives `MTL::Texture*`, never file paths.
- Materials/Environment/Lights/Accel → Integrator:
  `IntegratorSceneBindings` plus `WavefrontFeatures`.
- FrameUniforms → Integrator: FrameUniforms is the sole producer of per-frame
  `Uniforms`.
- Integrator → Post: radiance, AOVs, and guides. Post does not touch
  acceleration structures or materials.
- ScenePreparation orders `step*` calls but owns no domain resources.

Accel fills the `GeometryEntry` rows owned by Geometry while building BLASes.
It reads material flags to select opaque flags, instance options, and ray masks.

## Queue flow

The frame path is scene preparation → edits → CPU pose upload → Metal 4
skinning → acceleration-structure update → wavefront integration → post.

On Apple9+ the acceleration-structure update shares the Metal 4 queue with
skinning and tracing. Earlier GPUs use the Metal 3 AS queue through
`AccelBuildPath` and synchronize with a shared event. MetalFX denoising remains
on a Metal 3 post-process command buffer.

On the pre-Apple9 split, skinning is committed with
`Metal4Context::submitSkin()` on its own allocator ring, the Metal 3 build waits
on `skinEvent()`, and tracing waits on the build event. This must remain
GPU-chained: replacing it with submit-and-wait added about 19 ms of CPU blocking
per BrainStem frame and increased playback frames from roughly 100 ms to
300–600 ms.
