#ifndef STRELKA_BSDF_TYPES_H
#define STRELKA_BSDF_TYPES_H

// ============================================================================
// bsdf_types.h -- BSDF event types, sample results, and evaluation results
// ============================================================================

#include "material_math.h"

// NOLINTBEGIN(cppcoreguidelines-pro-type-member-init)
//
// Device-shared header: NVCC and the Metal compiler read this too, and
// clang-tidy only ever sees the host build, so these two suggestions cannot be
// taken here. Initialising the locals means a dead store in a BSDF inner loop --
// they are out-parameters written on the next line -- and the fixer spells the
// initialiser NAN, which needs <math.h>, which Metal rejects outright. Default
// member initialisers do the same to structs that are memcpy'd to the GPU.
// Suppressed rather than left to warn because these repeat in every translation
// unit that includes the header, and 700 lines of unactionable output per build
// is how the handful that matter get skipped.

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
// Result of bsdf_sample()
// ---------------------------------------------------------------------------
// The two results below use *different conventions for the cosine*, and mixing
// them up is silent:
//
//   bsdf_sample() -> bsdf_over_pdf  =  f * |cos(theta_i)| / pdf   (cosine included)
//   bsdf_eval()   -> bsdf           =  f                          (cosine NOT included)
//
// That is why a path tracer can write `throughput *= bsdf_over_pdf` after
// sampling, while next-event estimation has to multiply the cosine in itself --
// see connectLight()/connectEnvLight(), which fold it into the returned radiance.
//
// Multiple importance sampling weighs each strategy by the other's density, so
// the two routines must agree once put in the same convention:
//
//   bsdf_over_pdf  ==  bsdf * |cos(theta_i)| / pdf     (same wi, same si)
//
// tests/material/test_bsdf.cpp checks exactly that equality, and both pdfs, at
// the sampled direction. Comparing the two f's *without* converting looks like a
// failure of exactly the factor (1 - cos), which is a false alarm.
struct BsdfSampleResult
{
    float3          wi;         // Sampled incoming (light) direction, world space
    float           pdf;        // Probability density of wi (solid angle)
    float3          bsdf_over_pdf; // f(wo, wi) * |cos(theta_i)| / pdf
    unsigned int    event_type; // BsdfEventType flags
};

// GGX alpha at or below which a lobe is treated as a perfect mirror: sampled as
// a single direction, reported as a BSDF_EVENT_SPECULAR event, and given a
// discrete probability in the pdf field instead of a density.
//
// One constant because three separate places have to agree on it. The samplers
// use it to decide what kind of event they produced, bsdf_eval() uses it to
// decide it has nothing to evaluate, and bsdf_has_smooth_lobe() uses it to tell
// the integrator whether next-event estimation has anything to connect to at
// this vertex. alpha = roughness^2, so this is roughness 0.0316.
#define BSDF_DELTA_ALPHA 0.001f

// ---------------------------------------------------------------------------
// Result of bsdf_eval() -- evaluating the BSDF for a given direction pair
// ---------------------------------------------------------------------------
struct BsdfEvalResult
{
    float3  bsdf;   // f(wo, wi) -- the BSDF value, *without* the cosine
    float   pdf;    // Probability density of wi (solid angle)
};

#endif // STRELKA_BSDF_TYPES_H

// NOLINTEND(cppcoreguidelines-pro-type-member-init)
