#ifndef STRELKA_BSDF_TYPES_H
#define STRELKA_BSDF_TYPES_H

// ============================================================================
// bsdf_types.h -- BSDF event types, sample results, and evaluation results
// ============================================================================

#include "material_math.h"

// ---------------------------------------------------------------------------
// Event flags (bit-field, combinable with |)
// ---------------------------------------------------------------------------
enum BsdfEventType : unsigned int
{
    BSDF_EVENT_ABSORB              = 0u,
    BSDF_EVENT_DIFFUSE_REFLECTION  = (1u << 0),
    BSDF_EVENT_GLOSSY_REFLECTION   = (1u << 1),
    BSDF_EVENT_SPECULAR_REFLECTION = (1u << 2),
    BSDF_EVENT_DIFFUSE_TRANSMISSION  = (1u << 3),
    BSDF_EVENT_GLOSSY_TRANSMISSION   = (1u << 4),
    BSDF_EVENT_SPECULAR_TRANSMISSION = (1u << 5),

    // Convenience masks
    BSDF_EVENT_REFLECTION   = (BSDF_EVENT_DIFFUSE_REFLECTION |
                               BSDF_EVENT_GLOSSY_REFLECTION  |
                               BSDF_EVENT_SPECULAR_REFLECTION),
    BSDF_EVENT_TRANSMISSION = (BSDF_EVENT_DIFFUSE_TRANSMISSION |
                               BSDF_EVENT_GLOSSY_TRANSMISSION  |
                               BSDF_EVENT_SPECULAR_TRANSMISSION),
    BSDF_EVENT_SPECULAR     = (BSDF_EVENT_SPECULAR_REFLECTION |
                               BSDF_EVENT_SPECULAR_TRANSMISSION),
    BSDF_EVENT_GLOSSY       = (BSDF_EVENT_GLOSSY_REFLECTION |
                               BSDF_EVENT_GLOSSY_TRANSMISSION),
    BSDF_EVENT_DIFFUSE      = (BSDF_EVENT_DIFFUSE_REFLECTION |
                               BSDF_EVENT_DIFFUSE_TRANSMISSION),
    BSDF_EVENT_ALL          = (BSDF_EVENT_REFLECTION | BSDF_EVENT_TRANSMISSION),
};

// ---------------------------------------------------------------------------
// Delta event detection (specular reflection or transmission)
// ---------------------------------------------------------------------------
DEVICE_FUNC bool isDeltaEvent(unsigned int event_type)
{
    return (event_type & (BSDF_EVENT_SPECULAR_REFLECTION | BSDF_EVENT_SPECULAR_TRANSMISSION)) != 0;
}

// ---------------------------------------------------------------------------
// Result of bsdf_sample()
// ---------------------------------------------------------------------------
struct BsdfSampleResult
{
    float3          wi;         // Sampled incoming (light) direction, world space
    float           pdf;        // Probability density of wi (solid angle)
    float3          bsdf_over_pdf; // f(wo, wi) * |cos(theta_i)| / pdf
    unsigned int    event_type; // BsdfEventType flags
};

// ---------------------------------------------------------------------------
// Result of bsdf_eval() -- evaluating the BSDF for a given direction pair
// ---------------------------------------------------------------------------
struct BsdfEvalResult
{
    float3  bsdf;   // f(wo, wi) -- the BSDF value
    float   pdf;    // Probability density of wi (solid angle)
};

#endif // STRELKA_BSDF_TYPES_H
