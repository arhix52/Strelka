#ifndef STRELKA_MATERIAL_MATH_H
#define STRELKA_MATERIAL_MATH_H

// ============================================================================
// material_math.h -- Cross-platform math primitives for CUDA, Metal, and CPU
// ============================================================================

// STRELKA_MATERIAL_CUDA_HOST selects this branch for the OptiX backend's plain
// g++ translation units. They are host code, but they interoperate with device
// structs and already include CUDA's vector types and sutil, so they need the
// same spellings the device gets -- not the GLM ones the CPU branch installs.
// Taking the CPU branch there redefined make_float3/clamp/saturate on top of
// CUDA's and sutil's, which is what broke the Linux build.
#if defined(__CUDA_ARCH__) || defined(__CUDACC__) || defined(STRELKA_MATERIAL_CUDA_HOST)
// ---- CUDA (device code, nvcc host pass, and OptiX host code) ---------------
#    ifdef __CUDA_ARCH__
#        define DEVICE_FUNC __device__ __forceinline__
#    else
#        define DEVICE_FUNC inline
#    endif
// Storage class for a module-scope constant table; see sheen_albedo_lut.h.
//
// Under nvcc the table has to carry __device__ or it lands in host memory and
// every device function referencing it fails to resolve -- which is exactly
// what "identifier kSheenAlbedoLut is undefined" meant when the closest-hit
// module was compiled. `static` keeps it internal to the translation unit, so
// relocatable device code does not end up with duplicate definitions.
//
// Plain g++ building the OptiX host side reaches this branch too (see
// STRELKA_MATERIAL_CUDA_HOST) and does not know __device__, hence the split.
#    if defined(__CUDACC__)
#        define DEVICE_CONST static __device__ const
#    else
#        define DEVICE_CONST static const
#    endif
#    define THREAD_REF
#    define M_PI_F 3.14159265358979323846f
#    define M_1_PI_F 0.31830988618379067154f
#    define M_2_PI_F 0.63661977236758134308f
// float2, float3, float4 and make_float* are CUDA built-ins.
// dot, cross, normalize, length are available via sutil/vec_math.h.

#    ifndef STRELKA_MATERIAL_MATH_CUDA_FUNCS
#        define STRELKA_MATERIAL_MATH_CUDA_FUNCS
// saturate() is provided by sutil/vec_math_adv.h in CUDA builds
DEVICE_FUNC float sqr(float v)
{
    return v * v;
}

DEVICE_FUNC float3 mix(float3 a, float3 b, float t)
{
    return a + (b - a) * t;
}

DEVICE_FUNC float mix(float a, float b, float t)
{
    return a + (b - a) * t;
}

DEVICE_FUNC float luminance(float3 c)
{
    return 0.2126f * c.x + 0.7152f * c.y + 0.0722f * c.z;
}

DEVICE_FUNC float3 safe_normalize(float3 v)
{
    float len = length(v);
    return len > 1e-8f ? v / len : make_float3(0.0f, 1.0f, 0.0f);
}

DEVICE_FUNC float3 reflect_dir(float3 incident, float3 normal)
{
    return incident - 2.0f * dot(incident, normal) * normal;
}

#    endif

#elif defined(__METAL_VERSION__)
// ---- Metal Shading Language ------------------------------------------------
#    define DEVICE_FUNC inline
// Metal rejects a plain `const` array at module scope; it wants an explicit
// address space. See sheen_albedo_lut.h.
#    define DEVICE_CONST constant
#    define THREAD_REF thread
// M_PI_F is defined by <metal_stdlib>; only define if missing
#    ifndef M_PI_F
#        define M_PI_F 3.14159265358979323846f
#    endif
#    ifndef M_1_PI_F
#        define M_1_PI_F 0.31830988618379067154f
#    endif
#    ifndef M_2_PI_F
#        define M_2_PI_F 0.63661977236758134308f
#    endif

// C math compat aliases so shared headers (sampling.h, fresnel.h) compile
#    define sqrtf(x) metal::sqrt(x)
#    define cosf(x) metal::cos(x)
#    define sinf(x) metal::sin(x)
#    define fmaxf(x, y) metal::fmax(x, y)
#    define fminf(x, y) metal::fmin(x, y)
#    define fabsf(x) metal::fabs(x)
#    define fmodf(x, y) metal::fmod(x, y)
#    define copysignf(x, y) metal::copysign(x, y)
#    define acosf(x) metal::acos(x)
#    define asinf(x) metal::asin(x)
#    define tanf(x) metal::tan(x)
#    define atan2f(y, x) metal::atan2(y, x)
#    define expf(x) metal::exp(x)
#    define logf(x) metal::log(x)
#    define powf(x, y) metal::pow(x, y)
#    define fmaf(x, y, z) metal::fma(x, y, z)

inline float3 make_float3(float x, float y, float z)
{
    return float3(x, y, z);
}
inline float3 make_float3(float v)
{
    return float3(v);
}
inline float2 make_float2(float x, float y)
{
    return float2(x, y);
}
inline float4 make_float4(float x, float y, float z, float w)
{
    return float4(x, y, z, w);
}

// Use metal::saturate directly; do NOT define a wrapper (ambiguous with using namespace metal)
#    define saturate(v) metal::saturate(v)
inline float sqr(float v)
{
    return v * v;
}
inline float luminance(float3 c)
{
    return 0.2126f * c.x + 0.7152f * c.y + 0.0722f * c.z;
}

inline float3 safe_normalize(float3 v)
{
    float len = metal::length(v);
    return len > 1e-8f ? v / len : float3(0.0f, 1.0f, 0.0f);
}

inline float3 reflect_dir(float3 incident, float3 normal)
{
    return metal::reflect(incident, normal);
}

#else
// ---- CPU (tests, previews) ------------------------------------------------
#    define DEVICE_FUNC inline
#    define DEVICE_CONST static const
#    define THREAD_REF

#    include <glm/glm.hpp>
#    include <glm/gtc/constants.hpp>
#    include <cmath>
#    include <algorithm>

// Guarded with the same macro material_params.h uses, so whichever of the two
// is included first wins and the other is a no-op. They must agree on GLM --
// see the note at the top of material_params.h for what happens when they do
// not.
#    ifndef STRELKA_MATERIAL_FLOAT_TYPES
#        define STRELKA_MATERIAL_FLOAT_TYPES
using float2 = glm::vec2;
using float3 = glm::vec3;
using float4 = glm::vec4;
#    endif

// NOLINTBEGIN(modernize-return-braced-init-list)
//
// Naming the type is the whole point of a shim three compilers share, so a
// braced return would delete the only thing these lines say. Markers rather
// than trailing NOLINTs because clang-format splits a one-liner it is asked
// to format and carries the comment to the closing brace, where it suppresses
// nothing -- and this block is hand-aligned, so it is not formatted at all.
inline float3 make_float3(float x, float y, float z)
{
    return float3(x, y, z);
}
inline float3 make_float3(float v)
{
    return float3(v);
}
inline float2 make_float2(float x, float y)
{
    return float2(x, y);
}
inline float4 make_float4(float x, float y, float z, float w)
{
    return float4(x, y, z, w);
}
// NOLINTEND(modernize-return-braced-init-list)

using glm::cross;
using glm::dot;
using glm::length;
using glm::mix;
using glm::normalize;
using glm::reflect;
using glm::refract;

inline float clamp(float v, float lo, float hi)
{
    return std::max(lo, std::min(hi, v));
}

inline float saturate(float v)
{
    return clamp(v, 0.0f, 1.0f);
}
inline float sqr(float v)
{
    return v * v;
}

inline float luminance(float3 c)
{
    return 0.2126f * c.x + 0.7152f * c.y + 0.0722f * c.z;
}

inline float3 safe_normalize(float3 v)
{
    const float len = glm::length(v);
    return len > 1e-8f ? v / len : float3(0.0f, 1.0f, 0.0f);
}

#    define M_PI_F 3.14159265358979323846f
#    define M_1_PI_F 0.31830988618379067154f
#    define M_2_PI_F 0.63661977236758134308f

inline float3 reflect_dir(float3 incident, float3 normal)
{
    return incident - 2.0f * glm::dot(incident, normal) * normal;
}

#endif

// Float vectors can have finite components while dot(v,v) overflows. These
// helpers scale before squaring, and are shared by every backend so geometry
// support never depends on which compiler implements length().
DEVICE_FUNC float3 normalizeFiniteVectorOrZero(float3 v)
{
    const float scale = fmaxf(fabsf(v.x), fmaxf(fabsf(v.y), fabsf(v.z)));
    if (!(scale > 0.0f) || !(scale <= 3.402823466e38f))
    {
        return make_float3(0.0f);
    }
    const float3 scaled = v / scale;
    const float lengthSquared = dot(scaled, scaled);
    return lengthSquared > 0.0f ? scaled / sqrtf(lengthSquared) : make_float3(0.0f);
}

DEVICE_FUNC float finiteVectorLength(float3 v)
{
    const float scale = fmaxf(fabsf(v.x), fmaxf(fabsf(v.y), fabsf(v.z)));
    if (!(scale > 0.0f) || !(scale <= 3.402823466e38f))
    {
        return 0.0f;
    }
    const float3 scaled = v / scale;
    const float normalizedLength = sqrtf(dot(scaled, scaled));
    return normalizedLength <= 3.402823466e38f / scale ? scale * normalizedLength : 0.0f;
}


DEVICE_FUNC float3 finiteDirectionAndDistance(float3 offset, THREAD_REF float& distance)
{
    distance = finiteVectorLength(offset);
    return distance > 0.0f ? offset / distance : make_float3(0.0f);
}

DEVICE_FUNC float differenceOfProducts(float a, float b, float c, float d)
{
    const float cd = c * d;
    const float difference = fmaf(a, b, -cd);
    return difference + fmaf(-c, d, cd);
}

struct CompensatedFloat
{
    float high;
    float low;
};

DEVICE_FUNC CompensatedFloat compensatedSum(float a, float b)
{
    CompensatedFloat result{};
    result.high = a + b;
    const float virtualB = result.high - a;
    result.low = (a - (result.high - virtualB)) + (b - virtualB);
    return result;
}

DEVICE_FUNC CompensatedFloat compensatedProduct(float a, float b)
{
    CompensatedFloat result{};
    result.high = a * b;
    result.low = fmaf(a, b, -result.high);
    return result;
}

DEVICE_FUNC CompensatedFloat addCompensated(CompensatedFloat a, CompensatedFloat b)
{
    const CompensatedFloat highSum = compensatedSum(a.high, b.high);
    const CompensatedFloat lowSum = compensatedSum(a.low, b.low);
    const CompensatedFloat middle = compensatedSum(highSum.low, lowSum.high);
    const CompensatedFloat leading = compensatedSum(highSum.high, middle.high);
    const CompensatedFloat trailing = compensatedSum(leading.low, middle.low + lowSum.low);
    CompensatedFloat result = compensatedSum(leading.high, trailing.high);
    result.low += trailing.low;
    return result;
}

DEVICE_FUNC CompensatedFloat scaleCompensated(CompensatedFloat value, float scale)
{
    CompensatedFloat result = compensatedProduct(value.high, scale);
    const CompensatedFloat correction = compensatedSum(result.low, value.low * scale);
    const CompensatedFloat leading = compensatedSum(result.high, correction.high);
    result = compensatedSum(leading.high, leading.low + correction.low);
    return result;
}

DEVICE_FUNC CompensatedFloat multiplyCompensated(CompensatedFloat a, CompensatedFloat b)
{
    CompensatedFloat result = compensatedProduct(a.high, b.high);
    result = addCompensated(result, compensatedProduct(a.high, b.low));
    result = addCompensated(result, compensatedProduct(a.low, b.high));
    return addCompensated(result, compensatedProduct(a.low, b.low));
}

DEVICE_FUNC CompensatedFloat negateCompensated(CompensatedFloat value)
{
    value.high = -value.high;
    value.low = -value.low;
    return value;
}

DEVICE_FUNC CompensatedFloat compensatedDot3(CompensatedFloat ax,
                                             CompensatedFloat ay,
                                             CompensatedFloat az,
                                             CompensatedFloat bx,
                                             CompensatedFloat by,
                                             CompensatedFloat bz)
{
    return addCompensated(
        addCompensated(multiplyCompensated(ax, bx), multiplyCompensated(ay, by)), multiplyCompensated(az, bz));
}

DEVICE_FUNC float compensatedValue(CompensatedFloat value)
{
    return value.high + value.low;
}

DEVICE_FUNC CompensatedFloat divideCompensated(CompensatedFloat numerator, CompensatedFloat denominator)
{
    const float denominatorValue = compensatedValue(denominator);
    CompensatedFloat quotient = compensatedSum(compensatedValue(numerator) / denominatorValue, 0.0f);
    for (unsigned int iteration = 0u; iteration < 2u; ++iteration)
    {
        const CompensatedFloat residual =
            addCompensated(numerator, negateCompensated(multiplyCompensated(denominator, quotient)));
        quotient = addCompensated(quotient, compensatedSum(compensatedValue(residual) / denominatorValue, 0.0f));
    }
    return quotient;
}

DEVICE_FUNC CompensatedFloat sqrtCompensated(CompensatedFloat value)
{
    const float root = sqrtf(fmaxf(compensatedValue(value), 0.0f));
    CompensatedFloat result = compensatedSum(root, 0.0f);
    if (root > 0.0f)
    {
        const CompensatedFloat residual = addCompensated(value, negateCompensated(multiplyCompensated(result, result)));
        result = addCompensated(result, compensatedSum(compensatedValue(residual) / (2.0f * root), 0.0f));
    }
    return result;
}

struct ExactFloatExpansion
{
    float components[12];
    unsigned int size;
};

DEVICE_FUNC void addExactFloat(THREAD_REF ExactFloatExpansion& expansion, float value)
{
    float carry = value;
    unsigned int outputSize = 0u;
    for (unsigned int index = 0u; index < expansion.size; ++index)
    {
        const CompensatedFloat sum = compensatedSum(carry, expansion.components[index]);
        if (sum.low != 0.0f)
            expansion.components[outputSize++] = sum.low;
        carry = sum.high;
    }
    if (carry != 0.0f || outputSize == 0u)
        expansion.components[outputSize++] = carry;
    expansion.size = outputSize;
}

DEVICE_FUNC void addExactProduct(THREAD_REF ExactFloatExpansion& expansion, float a, float b)
{
    const CompensatedFloat product = compensatedProduct(a, b);
    if (product.low != 0.0f)
        addExactFloat(expansion, product.low);
    addExactFloat(expansion, product.high);
}

DEVICE_FUNC CompensatedFloat exactScalarTransmittedCosineSquared(float incidentCosine, float eta)
{
    // A product of two binary32 values is represented exactly by these two
    // terms. Expanding its square and eta^2 before the final subtraction keeps
    // the sign and magnitude even when the Snell remainder is below 2^-48 of
    // either operand.
    const CompensatedFloat etaCosine = compensatedProduct(eta, incidentCosine);
    ExactFloatExpansion expansion{};
    addExactProduct(expansion, etaCosine.high, etaCosine.high);
    addExactProduct(expansion, etaCosine.high, etaCosine.low);
    addExactProduct(expansion, etaCosine.high, etaCosine.low);
    addExactProduct(expansion, etaCosine.low, etaCosine.low);
    addExactProduct(expansion, -eta, eta);
    addExactFloat(expansion, 1.0f);

    CompensatedFloat result = compensatedSum(0.0f, 0.0f);
    for (unsigned int index = 0u; index < expansion.size; ++index)
        result = addCompensated(result, compensatedSum(expansion.components[index], 0.0f));
    return result;
}

DEVICE_FUNC CompensatedFloat dielectricTransmittedCosineSquared(CompensatedFloat incidentCosine, float eta)
{
    // 1 - eta^2 (1 - c^2) = (eta c)^2 - (eta - 1)(eta + 1).
    // This avoids first rounding two values near one and then subtracting them
    // at the critical angle. eta-1 is exact by Sterbenz for neighbouring media.
    const CompensatedFloat etaCosine = scaleCompensated(incidentCosine, eta);
    const CompensatedFloat etaSquaredCosineSquared = multiplyCompensated(etaCosine, etaCosine);
    const CompensatedFloat etaSquaredMinusOne =
        multiplyCompensated(compensatedSum(eta, -1.0f), compensatedSum(eta, 1.0f));
    const CompensatedFloat result = addCompensated(etaSquaredCosineSquared, negateCompensated(etaSquaredMinusOne));

    // Two-float arithmetic is ample until the two terms cancel to within one
    // part per million. Scalar sample-side cosines can then use the exact
    // binary32 expansion to certify both the TIR sign and the tiny root without
    // paying that cost for ordinary Fresnel evaluations.
    const float cancellationScale =
        fmaxf(fabsf(compensatedValue(etaSquaredCosineSquared)), fabsf(compensatedValue(etaSquaredMinusOne)));
    if (incidentCosine.low != 0.0f || fabsf(compensatedValue(result)) > 9.5367431640625e-7f * cancellationScale)
    {
        return result;
    }
    return exactScalarTransmittedCosineSquared(incidentCosine.high, eta);
}

DEVICE_FUNC bool refract_dir(float3 incident, float3 normal, float eta, float incidentCosineMagnitude, THREAD_REF float3& out)
{
    if (!(eta > 0.0f) || !(eta <= 3.402823466e38f))
        return false;

    incidentCosineMagnitude = saturate(incidentCosineMagnitude);
    const CompensatedFloat transmittedCosineSquared =
        dielectricTransmittedCosineSquared(compensatedSum(incidentCosineMagnitude, 0.0f), eta);
    if (!(compensatedValue(transmittedCosineSquared) > 0.0f))
        return false;

    const CompensatedFloat transmittedCosine = sqrtCompensated(transmittedCosineSquared);
    const float signedIncidentCosine = dot(normal, incident) < 0.0f ? -incidentCosineMagnitude : incidentCosineMagnitude;
    const CompensatedFloat normalScale = addCompensated(compensatedProduct(eta, signedIncidentCosine), transmittedCosine);
    const CompensatedFloat x =
        addCompensated(compensatedProduct(eta, incident.x), negateCompensated(scaleCompensated(normalScale, normal.x)));
    const CompensatedFloat y =
        addCompensated(compensatedProduct(eta, incident.y), negateCompensated(scaleCompensated(normalScale, normal.y)));
    const CompensatedFloat z =
        addCompensated(compensatedProduct(eta, incident.z), negateCompensated(scaleCompensated(normalScale, normal.z)));
    out = make_float3(compensatedValue(x), compensatedValue(y), compensatedValue(z));
    return true;
}

DEVICE_FUNC bool refract_dir(float3 incident, float3 normal, float eta, THREAD_REF float3& out)
{
    return refract_dir(incident, normal, eta, fabsf(dot(normal, incident)), out);
}

DEVICE_FUNC CompensatedFloat compensatedDifferenceOfProducts(float a, float b, float c, float d)
{
    CompensatedFloat negativeProduct = compensatedProduct(c, d);
    negativeProduct.high = -negativeProduct.high;
    negativeProduct.low = -negativeProduct.low;
    return addCompensated(compensatedProduct(a, b), negativeProduct);
}

DEVICE_FUNC CompensatedFloat compensatedDotCrossExpansion(float3 a, float3 b, float3 c)
{
    const CompensatedFloat x = scaleCompensated(compensatedDifferenceOfProducts(b.y, c.z, b.z, c.y), a.x);
    const CompensatedFloat y = scaleCompensated(compensatedDifferenceOfProducts(b.z, c.x, b.x, c.z), a.y);
    const CompensatedFloat z = scaleCompensated(compensatedDifferenceOfProducts(b.x, c.y, b.y, c.x), a.z);
    return addCompensated(addCompensated(x, y), z);
}

DEVICE_FUNC CompensatedFloat
compensatedDotCrossExpansion(CompensatedFloat ax, CompensatedFloat ay, CompensatedFloat az, float3 b, float3 c)
{
    const CompensatedFloat x = multiplyCompensated(compensatedDifferenceOfProducts(b.y, c.z, b.z, c.y), ax);
    const CompensatedFloat y = multiplyCompensated(compensatedDifferenceOfProducts(b.z, c.x, b.x, c.z), ay);
    const CompensatedFloat z = multiplyCompensated(compensatedDifferenceOfProducts(b.x, c.y, b.y, c.x), az);
    return addCompensated(addCompensated(x, y), z);
}

DEVICE_FUNC float compensatedDotCross(float3 a, float3 b, float3 c)
{
    return compensatedValue(compensatedDotCrossExpansion(a, b, c));
}

DEVICE_FUNC float3 accurateCross(float3 a, float3 b)
{
    return make_float3(differenceOfProducts(a.y, b.z, a.z, b.y), differenceOfProducts(a.z, b.x, a.x, b.z),
                       differenceOfProducts(a.x, b.y, a.y, b.x));
}

DEVICE_FUNC float accurateDot(float3 a, float3 b)
{
    return compensatedValue(addCompensated(
        addCompensated(compensatedProduct(a.x, b.x), compensatedProduct(a.y, b.y)), compensatedProduct(a.z, b.z)));
}

DEVICE_FUNC CompensatedFloat compensatedDotExpansion(float3 a, float3 b)
{
    return addCompensated(
        addCompensated(compensatedProduct(a.x, b.x), compensatedProduct(a.y, b.y)), compensatedProduct(a.z, b.z));
}

DEVICE_FUNC float decomposeFloatExponent(float value, THREAD_REF int& exponent)
{
#if defined(__METAL_VERSION__)
    return metal::frexp(value, exponent);
#else
    return frexpf(value, &exponent);
#endif
}

DEVICE_FUNC float scaleFloatExponent(float value, int exponent)
{
#if defined(__METAL_VERSION__)
    return metal::ldexp(value, exponent);
#else
    return ldexpf(value, exponent);
#endif
}

DEVICE_FUNC CompensatedFloat scaleCompensatedExponent(CompensatedFloat value, int exponent)
{
    return compensatedSum(scaleFloatExponent(value.high, exponent), scaleFloatExponent(value.low, exponent));
}

DEVICE_FUNC float3 finiteCrossDirection(float3 a, float3 b)
{
    const float scaleA = fmaxf(fabsf(a.x), fmaxf(fabsf(a.y), fabsf(a.z)));
    const float scaleB = fmaxf(fabsf(b.x), fmaxf(fabsf(b.y), fabsf(b.z)));
    if (!(scaleA > 0.0f) || !(scaleA <= 3.402823466e38f) || !(scaleB > 0.0f) || !(scaleB <= 3.402823466e38f))
    {
        return make_float3(0.0f);
    }
    int exponentA = 0;
    int exponentB = 0;
    decomposeFloatExponent(scaleA, exponentA);
    decomposeFloatExponent(scaleB, exponentB);
    const float3 scaledA = make_float3(
        scaleFloatExponent(a.x, -exponentA), scaleFloatExponent(a.y, -exponentA), scaleFloatExponent(a.z, -exponentA));
    const float3 scaledB = make_float3(
        scaleFloatExponent(b.x, -exponentB), scaleFloatExponent(b.y, -exponentB), scaleFloatExponent(b.z, -exponentB));
    return normalizeFiniteVectorOrZero(accurateCross(scaledA, scaledB));
}

// `numerator / length(cross(a, b))`, evaluated without first forming either
// the potentially overflowing cross product or its reciprocal. This is the
// useful quantity for area densities: the area itself need not fit in float as
// long as the final density does.
DEVICE_FUNC float finiteCrossReciprocal(float3 a, float3 b, float numerator)
{
    const float scaleA = fmaxf(fabsf(a.x), fmaxf(fabsf(a.y), fabsf(a.z)));
    const float scaleB = fmaxf(fabsf(b.x), fmaxf(fabsf(b.y), fabsf(b.z)));
    if (!(numerator > 0.0f) || !(numerator <= 3.402823466e38f) || !(scaleA > 0.0f) || !(scaleA <= 3.402823466e38f) ||
        !(scaleB > 0.0f) || !(scaleB <= 3.402823466e38f))
    {
        return 0.0f;
    }
    int exponentA = 0;
    int exponentB = 0;
    decomposeFloatExponent(scaleA, exponentA);
    decomposeFloatExponent(scaleB, exponentB);
    const float3 scaledA = make_float3(
        scaleFloatExponent(a.x, -exponentA), scaleFloatExponent(a.y, -exponentA), scaleFloatExponent(a.z, -exponentA));
    const float3 scaledB = make_float3(
        scaleFloatExponent(b.x, -exponentB), scaleFloatExponent(b.y, -exponentB), scaleFloatExponent(b.z, -exponentB));
    const float scaledLength = finiteVectorLength(accurateCross(scaledA, scaledB));
    if (!(scaledLength > 0.0f))
    {
        return 0.0f;
    }
    int numeratorExponent = 0;
    int lengthExponent = 0;
    const float numeratorMantissa = decomposeFloatExponent(numerator, numeratorExponent);
    const float lengthMantissa = decomposeFloatExponent(scaledLength, lengthExponent);
    const float reciprocal = scaleFloatExponent(
        numeratorMantissa / lengthMantissa, numeratorExponent - lengthExponent - exponentA - exponentB);
    return reciprocal > 0.0f && reciprocal <= 3.402823466e38f ? reciprocal : 0.0f;
}

DEVICE_FUNC float inverseFiniteCrossLength(float3 a, float3 b)
{
    return finiteCrossReciprocal(a, b, 1.0f);
}

#endif // STRELKA_MATERIAL_MATH_H
