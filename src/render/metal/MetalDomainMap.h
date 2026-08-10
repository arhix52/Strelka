#pragma once

// Domain map for the Metal path-tracer backend.
//
// Each component owns one graphics responsibility and its GPU resources.
// MetalRender is only the frame orchestrator (prepare → deform → integrate →
// post → present). Cross-domain access goes through explicit APIs / bindings,
// not through poking another module's members.
//
// Ownership (who may mutate):
//   MetalTextures          material maps (decode → mip/BC → MTL::Texture, disk cache)
//   MetalMaterials         GPU material table + cutout/medium/SSS scene flags
//   MetalGeometry          VB/IB, GeometryEntry, mesh records
//   MetalAccelStructure    BLAS/TLAS, instance buffer, motion geometry, scratch
//   MetalLights            analytic UniformLight buffer
//   MetalEnvironment       dome/IBL texture + alias table + envPdfScale
//   MetalSkinning          joint matrices, skin PSO, skinned VB writes
//   MetalFrameUniforms     camera / jitter / exposure / SHARC → Uniforms
//   MetalWavefrontIntegrator  path queues, variant PSO cache, encode
//   MetalPostProcess       tonemap, guide/display/upscale textures (MetalFxContext)
//   MetalScenePreparation  BuildStage state machine only (no GPU resource ownership)
//
// Contracts:
//   Geometry → Accel: Accel descriptors name VB offsets/counts only; Skinning
//     writes VB contents and must not rebuild descriptors. Refit/rebuild via Accel.
//   Textures → Materials: Materials receives MTL::Texture*, never file paths.
//   Materials/Env/Lights/Accel → Integrator: IntegratorSceneBindings + WavefrontFeatures.
//   FrameUniforms → Integrator: sole producer of per-frame Uniforms.
//   Integrator → Post: radiance / AOV / guides; Post does not touch AS or materials.
//   ScenePreparation: orders step* calls; does not own domain resources.
//
// Already separate: Metal4Context, MetalFxContext, MetalBuffer.
//
// Pure (no Metal.hpp) helpers live next to their domain:
//   sampling_math.h, ibl_alias_table.h, integrator_features.h, texture_cache_key.h,
//   integrator_buffer_sizes.h
//
// Accel fills GeometryEntry rows via MetalGeometry::geometryEntries() while
// building BLASes (Geometry owns the table; Accel fills it). Accel reads
// Materials::isCutout() / hasAlphaMaterials() / isMediumBoundary() for opaque
// flags, instance options, and ray masks.
//
// Frame flow in MetalRender::render(): preparation → scene edits → skinning →
// accel pose update → FrameUniforms::fill → integrator → post.
