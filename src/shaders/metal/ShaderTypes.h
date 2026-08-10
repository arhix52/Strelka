#ifndef ShaderTypes_h
#define ShaderTypes_h

#include <simd/simd.h>
#ifndef __METAL_VERSION__
#ifdef __cplusplus
#include <Metal/MTLTypes.hpp>
#endif
#endif

#define GEOMETRY_MASK_TRIANGLE 1
#define GEOMETRY_MASK_CURVE 2
#define GEOMETRY_MASK_LIGHT 4
// A light the camera must not see directly but that still lights the scene and
// still appears in reflections. V-Ray calls it "invisible"; it is how a softbox
// stays out of frame while doing its job. Its own bit rather than a per-light
// test in the shader, because the distinction is exactly what a ray mask is for.
#define GEOMETRY_MASK_LIGHT_HIDDEN 8
// The boundary of a participating medium. Its own bit because a shadow ray must
// not be stopped by it -- RAY_MASK_SHADOW is the geometry bits alone, so a fog
// gizmo left on the triangle mask would black out everything it encloses.
#define GEOMETRY_MASK_MEDIUM 16

#define GEOMETRY_MASK_GEOMETRY (GEOMETRY_MASK_TRIANGLE | GEOMETRY_MASK_CURVE)

#define RAY_MASK_PRIMARY (GEOMETRY_MASK_GEOMETRY | GEOMETRY_MASK_LIGHT | GEOMETRY_MASK_MEDIUM)
#define RAY_MASK_SHADOW GEOMETRY_MASK_GEOMETRY
#define RAY_MASK_SECONDARY (RAY_MASK_PRIMARY | GEOMETRY_MASK_LIGHT_HIDDEN)

#ifndef __METAL_VERSION__
struct packed_float3
{
#    ifdef __cplusplus
    packed_float3() = default;
    packed_float3(vector_float3 v) : x(v.x), y(v.y), z(v.z)
    {
    }
#    endif
    float x;
    float y;
    float z;
};
#endif

enum class DebugMode : uint32_t
{
    eNone = 0,
    eNormal,
    eMotionBlur,
    // Denoiser guides, visualised by the resolve pass. Selecting any of these
    // turns their production on, so they cost nothing when not being looked at.
    eAovDiffuseAlbedo,
    eAovSpecularAlbedo,
    eAovNormal,
    eAovRoughness,
    eAovDepth,
    eAovMotion,
    // The two guides that had no view of their own, which is how a mask covering
    // a fifth of the frame went unnoticed until an object disappeared.
    eAovReactive,
    eAovSpecularHitDistance,
};

#define DEBUG_MODE_FIRST_AOV 3

struct Vertex
{
    vector_float3 pos;
    uint32_t tangent;

    uint32_t normal;
    uint32_t uv;
    float pad0;
    float pad1;
};

struct Uniforms
{
    simd::float4x4 viewToWorld;
    simd::float4x4 clipToView;
    simd::float4x4 prevViewToWorld;
    simd::float4x4 prevClipToView;
    vector_float3 missColor;

    uint32_t width;
    uint32_t height;
    uint32_t frameIndex;
    uint32_t subframeIndex;

    uint32_t numLights;
    uint32_t enableAccumulation;
    uint32_t samples_per_launch;
    uint32_t maxDepth;

    uint32_t rectLightSamplingMethod;
    // VOLUME_MODEL_GLTF or VOLUME_MODEL_CYCLES; see volume.h for why this is a
    // setting rather than a constant.
    uint32_t volumeModel;
    // 0 - Halton, 1 - PCG, 2 - Sobol (Owen), 3 - Sobol + blue noise, 4 - hybrid
    uint32_t samplerType;
    /// Sample count at which sampler 4 hands the frame from the blue-noise
    /// sequence to the per-pixel scrambled one.
    uint32_t blueNoiseSwitchSpp;

    uint32_t tonemapperType; // 0 - "None", "Reinhard", "ACES", "Filmic"
    float gamma; // 0 - off
    vector_float3 exposureValue;

    uint32_t debug;

    uint32_t enableMotionBlur;
    uint32_t isMotionBlurVisible;
    uint32_t enableCameraMotionBlur;

    // Depth of field
    int32_t useDof;
    float focalDistance;
    float lensRadius;
    int32_t apertureBlades;
    float bladeRotation;
    float anamorphicRatio;

    // Lens shift
    float shiftX;
    float shiftY;

    // Projection: 0 = perspective, 1 = orthographic. An orthographic camera is
    // not expressible as a clipToView matrix the perspective path can share --
    // it has no centre of projection, so the ray origin varies across the film
    // and the direction does not -- hence a flag and the film half-extents
    // rather than a different matrix.
    // Values are Camera::ProjectionType; PROJECTION_* below names them for the
    // shaders, which cannot see the host enum.
    uint32_t projectionType;
    float orthoHalfWidth;
    float orthoHalfHeight;

    // Environment map (dome light)
    uint32_t hasEnvMap;
    // Atmospheric scattering, homogeneous below fogHeight. See fog.h for why a
    // slab and not a bounded volume.
    uint32_t hasFog;
    /// Whether any material in the scene bounds a medium. Gates the extra
    /// traversal the shadow stage needs to attenuate through one, so a scene with
    /// only subsurface media -- whose boundaries a shadow ray never crosses --
    /// pays nothing for it.
    uint32_t hasBoundedMedium;
    // Radiance cache; see sharc.h.
    uint32_t sharcCapacity;   // 0 disables
    uint32_t sharcMinSamples; // before a voxel may be read
    uint32_t sharcDepth;      // first bounce allowed to read the cache
    float sharcBaseSize;
    float fogSigmaT;
    float fogAnisotropy;
    float fogHeight;
    vector_float3 fogAlbedo;
    // A separate environment for camera rays. See Scene::EnvLightDesc.
    uint32_t hasEnvBackground;
    float envBackgroundIntensity;
    uint32_t envMapWidth;
    uint32_t envMapHeight;
    float envMapIntensity;
    float envMapRotation;
    // (w*h) / (2*pi^2 * totalPower): converts a texel's luminance straight into
    // its solid-angle sampling PDF, so no CDF or PDF table is needed on the GPU.
    float envPdfScale;
    // Validation switches. estimatorMode: 0 = NEE + MIS (normal), 1 = BSDF
    // sampling only. The two are independent unbiased estimators of the same
    // integral, so at convergence they must produce the same image; the
    // difference between them measures estimator inconsistency directly.
    uint32_t estimatorMode;
    // Ray mask for camera/secondary rays. Excluding light geometry is the only
    // way to actually remove analytic lights: zeroing numLights just disables
    // NEE's light selection, the emissive geometry is still hit by BSDF rays.
    uint32_t primaryRayMask;
    vector_float3 envMapColorTint;
    // Previous frame's world-to-clip, for reprojecting a hit point into the last
    // frame's screen space. The motion-blur matrices are the inverses and cannot
    // be used for this.
    simd::float4x4 prevWorldToClip;
    // This frame's world-to-clip. Needed for the device-depth guide, which cannot
    // be derived from the inverses the camera-ray path carries.
    simd::float4x4 worldToClip;
    uint32_t writeAov;
    // Which convention the depth guide is written in; see kDenoiseDepth* below.
    uint32_t denoiseDepthMode;
    // Whether the previous frame's pose and instance transforms are valid to
    // read. False on the first frame of a scene and after anything that
    // invalidates the correspondence between frames.
    uint32_t hasPrevFramePose;
    /// Feed the denoiser the accumulated mean instead of this launch's samples.
    uint32_t useAccumulatedColor;
    /// Luminance ceiling, in exposed units, for the colour handed to the
    /// denoiser. Zero disables it.
    float denoiseFireflyClamp;
    /// Upper bound on what one indirect path may contribute; 0 disables it.
    ///
    /// Separate from denoiseFireflyClamp, which only conditions the denoiser's
    /// input and leaves the rendered image and the EXR alone. This one changes
    /// the image, so it is off by default -- see clampIndirectContribution.
    float clampIndirect;
    // Sub-pixel offset applied to every pixel of this frame, in pixels. Temporal
    // upscaling needs the whole image shifted by a known amount it can undo; the
    // per-pixel random jitter that antialiases a still frame is noise to it.
    float jitterX;
    float jitterY;
    uint32_t useFrameJitter;
    /// When set, sample zero is a deterministic shutter-close guide pass. Its
    /// radiance is discarded and the remaining samples estimate the image.
    uint32_t canonicalGuideSample;
    /// Light candidates drawn per shading point before one is resampled -- the
    /// M of resampled importance sampling. One is plain next-event estimation
    /// and the arithmetic reduces to exactly what it was.
    uint32_t risCandidates;
    /// 0 = sample level 0 (what a compute kernel does by default), 1 = ray-cone
    /// level of detail. A switch rather than a constant because the whole point
    /// of it is a memory-pressure trade that has to be measured per scene.
    uint32_t textureLodMode;
    /// Where the denoiser's material guides come from. 0 = walk to the first
    /// surface rough enough to describe, so that a mirror hands over the world
    /// it reflects rather than its own featureless albedo. 1 = the primary hit,
    /// always.
    ///
    /// A switch because the two are right for different things and neither is
    /// right for both: the walk is what makes a reflection denoisable, and it is
    /// also what makes a glossy floor's guides flicker between the floor and
    /// whatever it reflects, one pixel to the next.
    uint32_t guidePrimaryHit;
};



// How the depth guide is encoded.
//
// MetalFX does not document which it wants. Two things point at device depth: the
// scaler takes a viewToClipMatrix, which is only useful for undoing a projection,
// and depthReversed defaults to YES, which is a statement about NDC. Against that,
// the texture is R32Float and a linear distance would fit it. So the renderer can
// write any of the three and the choice is settled by measurement rather than by
// reading the header harder.
// Uniforms::projectionType. Mirrors oka::Camera::ProjectionType, which the
// shaders cannot include.
#define PROJECTION_PERSPECTIVE  0u
#define PROJECTION_ORTHOGRAPHIC 1u

#define kDenoiseDepthDevice 0u ///< clip z / w, the value a depth buffer holds
#define kDenoiseDepthViewZ  1u ///< distance along the camera's forward axis
#define kDenoiseDepthRadial 2u ///< distance to the eye

// What a denoiser needs to know about the primary hit, written once per pixel by
// the stage that shades it (or by the miss stage for background).
//
// One packed record in a buffer rather than six render targets: `shade` is the
// most register-pressured kernel in the tracer and a single buffer write costs it
// far less than binding and writing six textures. A resolve pass afterwards
// spreads it into the texture formats MetalFX wants, which keeps every format
// decision in one place.
struct AovSample
{
    packed_float3 diffuseAlbedo;
    float depth;            // encoding selected by Uniforms::denoiseDepthMode
    packed_float3 specularAlbedo;
    float roughness;
    packed_float3 normal;   // world space
    float motionX;          // previous-frame screen position minus current, in pixels
    float motionY;
    /// Distance from the primary hit to what its specular lobe sees. MetalFX
    /// reprojects reflections with this instead of treating them as if they sat
    /// on the surface. Zero when the surface is not specular.
    float specularHitDistance;
    /// 0 = trust the history here, 1 = ignore it. Raised where the motion vector
    /// is known to be a lie: mirrors, glass, and anything whose previous position
    /// could not be established.
    float reactive;
    float pad2;
};

struct UniformsTonemap
{
    // Render resolution: the linear radiance buffer is indexed with it.
    uint32_t width;
    uint32_t height;
    // Display resolution. Not the same thing once anything upscales, and the
    // texture-input tonemapper covers this, not the render size.
    uint32_t outWidth;
    uint32_t outHeight;


    uint32_t tonemapperType; // 0 - "None", "Reinhard", "ACES", "Filmic"
    float gamma; // 0 - off
    float maxEDR;
    vector_float3 exposureValue;
};

// Per-primitive attributes stored inside the acceleration structure.
//
// positions must be packed: a 16-byte-aligned float3 would pad the array to 48
// bytes and the struct to 96, where the packed form needs 36 and 72. On a
// skinned mesh this buffer is also rewritten by the skinning pass every frame,
// so the padding cost both memory and bandwidth for nothing.
struct Triangle
{
    packed_float3 positions[3];
    uint32_t normals[3];
    uint32_t tangent[3];
    uint32_t uv[3];
};

// Per-geometry data, indexed by (instance userID + intersection.geometry_id).
//
// One acceleration structure now holds many geometries — a glTF mesh's
// primitives are split by material, and merging them into a single BLAS is what
// keeps the top-level structure small. The material therefore can no longer
// travel in the instance's userID; userID holds the instance's base offset into
// this table instead, and the geometry index within the BLAS selects the entry.
struct GeometryEntry
{
    uint32_t vbOffset;     // mesh vertex buffer offset, or first control point of a curve set
    uint32_t indexOffset;  // mesh index buffer offset, or first segment of a curve set
    uint32_t materialId;
    // Zero for a triangle mesh. For a curve set: GEOM_FLAG_CURVE, plus the
    // segments per strand in the low bits, which is what lets the shader recover
    // root-to-tip position from the segment index alone -- a strand's own
    // parameter, for no memory at all. Zero there means the set has strands of
    // differing lengths and the gradient is not available.
    uint32_t flags;
};

#define GEOM_FLAG_CURVE        (1u << 31)
#define GEOM_CURVE_CUBIC       (1u << 30)
#define GEOM_CURVE_STRAND_MASK 0x0000FFFFu

// --- Wavefront path tracing ------------------------------------------------
//
// The megakernel keeps a path's state in registers, which is free but forces
// every lane of a simdgroup to wait for the longest-lived path in it. The
// wavefront tracer trades that for explicit state in memory, so each stage only
// runs over paths that are still alive. Memory traffic is therefore the design
// constraint, and this struct is deliberately kept at 48 bytes.
//
// Two things are *not* stored:
//   - the sampler, because it is a pure function of
//     (pixelIndex, sampleIndex, depth) and is cheaper to recompute than to load;
//   - the IOR stack (36 B), which only matters to paths currently inside a
//     dielectric and lives in a side table indexed by path slot.
// The ray is separate from the rest of the state because `extend` reads only
// the ray and is the most traffic-sensitive stage: keeping them together made it
// pull 48 bytes per path to use 24. `shade` reads both, so nothing is read twice.
//
// There is no pixelIndex: a path lives in the slot of the pixel it belongs to,
// so its index *is* its pixel.
struct PathRay
{
    packed_float3 origin;
    packed_float3 direction;
};

struct PathState
{
    packed_float3 throughput;
    uint32_t depthAndFlags; // depth in bits 0..7, flags above
    float lastBsdfPdf;
    // How far the ray has travelled since the vertex `lastBsdfPdf` was measured
    // at. Zero for every path that has not passed through anything.
    //
    // Passing through a cutout resets the ray's origin to the surface it slipped
    // past, and the multiple-importance weight at an area light needs the
    // distance from the vertex that *scattered*, not from wherever the ray was
    // last restarted. The direction does not change across a pass-through, so
    // one number recovers that vertex: origin - direction * this.
    float misDistance;

    // Radiance cache bookkeeping. A path that passes through a cache voxel
    // remembers the slot, what the pixel had already gathered at that moment and
    // the reciprocal of its throughput there; when the path ends, the difference
    // over that throughput is what the rest of the path was worth from that
    // voxel, and that is what the cache stores.
    //
    // Costs 28 bytes on every live path and buys the whole tail of the path, so
    // it is only allocated when the cache is on.
    uint32_t sharcIndex;
    packed_float3 sharcRadianceAtVisit;
    packed_float3 sharcInvThroughput;

    /// Which participating medium the path is inside, and how many scattering
    /// events it has had there: material index + 1 in the low 16 bits, step count
    /// in the high 16. Zero means the path is outside every medium.
    ///
    /// One slot, so media do not nest: a path inside a fog volume that enters a
    /// block of wax takes the wax and forgets the fog until it leaves. Nesting
    /// needs a stack like the one the dielectrics keep, and nothing in the scenes
    /// this serves overlaps two media.
    ///
    /// One packed word rather than the medium's parameters, because this is per
    /// pixel and the parameters are per material: at 1024x1024 carrying sigma_t,
    /// albedo and g would cost 28 MB to avoid a load from a table that fits in
    /// cache.
    uint32_t medium;
    /// The medium's single-scattering albedo at the point the path entered it,
    /// packed RGBA8.
    ///
    /// Carried rather than read from the material at each scattering event,
    /// because inside a medium there is no surface left to sample a texture on:
    /// the only place the marble's veining exists is the boundary the walk came
    /// through. Packed to one word -- a scattering albedo has nothing like eight
    /// bits of meaningful precision, and this is per pixel.
    uint32_t mediumAlbedo;
};

#define MEDIUM_INDEX_MASK  0xFFFFu
#define MEDIUM_STEP_SHIFT   16u
/// Ceiling on one walk. A dense medium is a long walk that Russian roulette
/// alone terminates slowly, and a path that never ends is a hang rather than a
/// dim pixel.
#define MEDIUM_MAX_STEPS    256u

#define SHARC_NO_ENTRY 0xFFFFFFFFu


#define PATH_FLAG_ALIVE      (1u << 8)
#define PATH_FLAG_SPECULAR   (1u << 9)
#define PATH_FLAG_NEE_DONE   (1u << 10)
// The denoiser guides for this pixel have been written. A mirror or a glass
// surface has no albedo to demodulate against and a roughness of nothing, so the
// guides are deferred to the first surface that does -- and then must not be
// overwritten by the bounce after it.
#define PATH_FLAG_AOV_DONE   (1u << 11)
#define PATH_DEPTH_MASK      0xFFu
// Transparent hits are counted apart from bounces: passing through a cutout is
// not a scattering event and must not consume path depth. Bits 12+ are free.
#define PATH_PASSTHROUGH_SHIFT 12u
#define PATH_PASSTHROUGH_MAX   32u

// What `extend` hands to `shade`. Deliberately small: `intersection.primitive_data`
// is only valid inside the kernel that ran the intersect, so instead of copying
// vertex attributes across, `shade` refetches them from the vertex buffer using
// the geometry entry — the same lookup the motion-blur path already performs.
struct HitRecord
{
    uint32_t geomEntryIndex; // instance userID + intersection.geometry_id
    // The TLAS instance that was hit. Carried rather than looked up from the
    // geometry entry, because a BLAS is shared by every instance of the same
    // object -- 38 000 scattered trees over 50 distinct meshes in the pine
    // forest -- and the geometry it holds therefore belongs to no single
    // instance. The intersection knows which one; nothing downstream does.
    uint32_t instanceIndex;
    uint32_t primitiveId;
    vector_float2 barycentrics; // not float2: this header is compiled by the host too
    float distance; // < 0 means the ray escaped
};

// A deferred occlusion query produced by `shade` and consumed by `shadow`.
struct ShadowRay
{
    packed_float3 origin;
    packed_float3 direction;
    packed_float3 weight; // radiance already divided by pdf and multiplied by the BSDF
    float maxDistance;
    uint32_t pixelIndex;
    // Threshold at which traversal may give up on this ray, drawn where the ray
    // was created because that is where the sampler knows the path's depth.
    float rrCutoff;
    /// Which bounded medium the ray starts inside, material index + 1, or 0.
    ///
    /// Carried rather than re-derived: the shadow stage can find where the ray
    /// *leaves* a medium by tracing its boundary, but nothing in the ray itself
    /// says whether it began within one. A vertex inside a fog volume and a
    /// vertex just outside it produce the same origin and direction.
    uint32_t medium;
};

// One entry of the environment map alias table (Walker/Vose), one per texel.
// Sampling is a single load: draw a bucket uniformly, then keep it with
// probability `prob`, otherwise jump to `alias`.
struct EnvAliasEntry
{
    float prob;
    uint32_t alias;
};

struct SkinningParams
{
    uint32_t vbOffset;       // vertex buffer offset for this mesh
    uint32_t sbOffset;       // skin data buffer offset
    uint32_t jointMatOffset; // offset into joint matrices array
    uint32_t vertexCount;
};

struct TriangleUpdateParams
{
    uint32_t triangleCount;
    uint32_t indexOffset;    // mesh.mIndex
    uint32_t vbOffset;       // mesh.mVbOffset
    uint32_t pad0;
};

// GPU side structure
// pad0: spot inner cone (rad) or point soft radius.
// pad1: KHR attenuation range (0 = infinite).
struct UniformLight
{
    vector_float4 points[4];
    vector_float4 color;
    vector_float4 normal;
    int type;
    float halfAngle;
    float pad0;
    float pad1;
};

struct Material
{
    // PBR parameters (layout uses packed_float3 for host/GPU compatibility)
    packed_float3 base_color;       // 12 bytes
    float metallic;                 //  4 bytes  -- 16

    float roughness;                //  4 bytes
    float ior;                      //  4 bytes
    float specular;                 //  4 bytes
    float _pad_specular;            //  4 bytes  -- 32 (was specular_tint)

    float transmission;             //  4 bytes
    float clearcoat;                //  4 bytes
    float clearcoat_roughness;      //  4 bytes
    float anisotropy;               //  4 bytes  -- 48

    packed_float3 emission;         // 12 bytes
    float emission_strength;        //  4 bytes  -- 64

    float normal_scale;             //  4 bytes
    float occlusion_strength;       //  4 bytes
    float alpha_cutoff;             //  4 bytes
    uint32_t material_type;         //  4 bytes  -- 80

    uint32_t thin_walled;           //  4 bytes
    uint32_t dielectric_priority;   //  4 bytes  (nested dielectrics)
    uint32_t alpha_mode;            //  4 bytes  (AlphaMode)
    float base_color_alpha;         //  4 bytes  -- 96

    packed_float3 attenuation_color; // 12 bytes (KHR_materials_volume)
    float attenuation_distance;      //  4 bytes -- 112

    // KHR_texture_transform, one per material; see material_params.h.
    vector_float2 uv_offset;         //  8 bytes
    vector_float2 uv_scale;          //  8 bytes
    float uv_rotation;               //  4 bytes
    float _pad_uv;                   //  4 bytes -- 136

    // KHR_materials_diffuse_transmission; see material_params.h.
    packed_float3 diffuse_transmission_color; // 12 bytes
    float diffuse_transmission;               //  4 bytes -- 152

    // KHR_materials_sheen; see material_params.h.
    packed_float3 sheen_color;                // 12 bytes
    float sheen;                              //  4 bytes -- 168

    // STRELKA_materials_subsurface; see material_params.h.
    packed_float3 subsurface_radius;          // 12 bytes
    float sheen_roughness;                    //  4 bytes -- 184
    float subsurface;                         //  4 bytes
    float subsurface_anisotropy;              //  4 bytes -- 192

    // STRELKA_materials_medium; see material_params.h.
    packed_float3 medium_emission;            // 12 bytes
    uint32_t medium_flags;                    //  4 bytes -- 208
    float clearcoat_ior;                      //  4 bytes -- 212

    // KHR_materials_specular specularColorFactor; see material_params.h.
    packed_float3 specular_color;             // 12 bytes -- 224

    // KHR_materials_iridescence; see material_params.h.
    float iridescence;                        //  4 bytes
    float iridescence_ior;                    //  4 bytes
    float iridescence_thickness;              //  4 bytes -- 236

    packed_float3 subsurface_reference;       // 12 bytes -- 248
    float _pad_irid[2];                       //  8 bytes -- 256

    // Textures (8 bytes each: resource ID on CPU, texture handle on GPU)
#ifdef __METAL_VERSION__
    texture2d<float> baseColorTexture;
    texture2d<float> metallicRoughnessTexture;
    texture2d<float> normalTexture;
    texture2d<float> emissionTexture;
    texture2d<float> occlusionTexture;
#else
    MTL::ResourceID baseColorTexture;
    MTL::ResourceID metallicRoughnessTexture;
    MTL::ResourceID normalTexture;
    MTL::ResourceID emissionTexture;
    MTL::ResourceID occlusionTexture;
#endif
};

#endif
