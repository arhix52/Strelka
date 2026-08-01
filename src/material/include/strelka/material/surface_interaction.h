#ifndef STRELKA_SURFACE_INTERACTION_H
#define STRELKA_SURFACE_INTERACTION_H

// ============================================================================
// surface_interaction.h -- Geometric and shading data at the hit point
// ============================================================================

#include "material_math.h"

// ---------------------------------------------------------------------------
// SurfaceInteraction -- everything the BSDF needs about the surface hit
//
// Filled in by the closest-hit program (or intersection code) before the
// material is evaluated.  All vectors are in world space.
// ---------------------------------------------------------------------------
struct SurfaceInteraction
{
    // -- Position -----------------------------------------------------------
    float3  position;           // World-space hit point

    // -- Geometry -----------------------------------------------------------
    float3  geometry_normal;    // Face (geometric) normal, normalized
    float3  shading_normal;     // Shading normal (may differ from geom normal)
    float3  tangent;            // Shading tangent  (T)
    float3  bitangent;          // Shading bitangent (B = N x T) or from mesh

    // -- Texture coordinates ------------------------------------------------
    float2  uv;                 // Primary UV set

    // -- View direction -----------------------------------------------------
    float3  wo;                 // Outgoing direction (toward camera), world space

    // -- Material -----------------------------------------------------------
    // Pointer (or index) to the material params is stored externally; these
    // resolved values are computed *after* texture lookups.
    float3  albedo;             // Resolved base color (texture * vertex color * param)
    float   metallic;           // Resolved metallic
    float   roughness;          // Resolved roughness (clamped to [min, 1])
    float   ior;                // Index of refraction
    float   transmission;       // Transmission weight [0, 1]
    float3  emission;           // Resolved emission color * strength
    float   clearcoat;          // Clearcoat weight
    float   clearcoat_roughness;// Clearcoat roughness
    float   anisotropy;         // Anisotropy [-1, 1]
    float   specular;           // Specular level
    float   specular_tint;      // Specular tint

    unsigned int material_type; // MaterialType tag
    unsigned int thin_walled;   // 1 = thin-walled surface
    unsigned int dielectric_priority; // Priority for nested dielectrics (0 = air)

    // -- Nested dielectrics -------------------------------------------------
    float   exterior_ior;       // IOR of the medium the ray is traveling through

    // -- Flags / state ------------------------------------------------------
    bool    front_face;         // True when the ray hit the front face
};

#endif // STRELKA_SURFACE_INTERACTION_H
