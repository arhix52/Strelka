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

#define GEOMETRY_MASK_GEOMETRY (GEOMETRY_MASK_TRIANGLE | GEOMETRY_MASK_CURVE)

#define RAY_MASK_PRIMARY (GEOMETRY_MASK_GEOMETRY | GEOMETRY_MASK_LIGHT)
#define RAY_MASK_SHADOW GEOMETRY_MASK_GEOMETRY
#define RAY_MASK_SECONDARY GEOMETRY_MASK_GEOMETRY

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
};

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
    uint32_t samplerType; // 0 - Halton, 1 - PCG

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

    // Environment map (dome light)
    uint32_t hasEnvMap;
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
};

struct UniformsTonemap
{
    uint32_t width;
    uint32_t height;
    
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
    uint32_t vbOffset;     // mesh vertex buffer offset
    uint32_t indexOffset;  // mesh index buffer offset
    uint32_t materialId;
    // Emitted TLAS instance this geometry belongs to. The megakernel reads the
    // object-to-world transform straight off the intersection, but the wavefront
    // tracer shades in a separate kernel where the intersection is gone, so it
    // needs a way back to the instance descriptor. Each emitted instance owns a
    // contiguous run of geometry entries, so this is well defined.
    uint32_t instanceIndex;
};

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
struct PathState
{
    packed_float3 origin;
    packed_float3 direction;
    packed_float3 throughput;
    uint32_t pixelIndex;
    uint32_t depthAndFlags; // depth in bits 0..7, flags above
    float lastBsdfPdf;
};

#define PATH_FLAG_ALIVE      (1u << 8)
#define PATH_FLAG_SPECULAR   (1u << 9)
#define PATH_FLAG_NEE_DONE   (1u << 10)
#define PATH_DEPTH_MASK      0xFFu

// What `extend` hands to `shade`. Deliberately small: `intersection.primitive_data`
// is only valid inside the kernel that ran the intersect, so instead of copying
// vertex attributes across, `shade` refetches them from the vertex buffer using
// the geometry entry — the same lookup the motion-blur path already performs.
struct HitRecord
{
    uint32_t geomEntryIndex; // instance userID + intersection.geometry_id
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
    float specular_tint;            //  4 bytes  -- 32

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
    uint32_t _pad1;                 //  4 bytes
    uint32_t _pad2;                 //  4 bytes  -- 96

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
