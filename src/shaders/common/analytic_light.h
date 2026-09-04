#ifndef STRELKA_ANALYTIC_LIGHT_H
#define STRELKA_ANALYTIC_LIGHT_H

// Shared analytic disc/ellipsoid measure and intersection math. The axes are
// the columns of the authored object-to-world linear transform, including the
// light radius. Sampling and intersection therefore describe the same smooth
// geometry even under shear, non-uniform scale, or a mirrored transform.

#include <strelka/material/material_math.h>
#include <light_types.h>

enum : unsigned int
{
    STRELKA_ANALYTIC_LIGHT_CAMERA_BIT = 1u,
    STRELKA_ANALYTIC_LIGHT_SECONDARY_BIT = 2u
};

DEVICE_FUNC bool lightUsesAnalyticAreaIntersection(int lightType)
{
    return lightType == LIGHT_TYPE_DISC || lightType == LIGHT_TYPE_SPHERE;
}

struct AnalyticLightSample
{
    float3 point{};
    float3 normal{};
    // Density with respect to world area. Keeping p_A rather than 1/p_A is
    // essential when a very large finite surface has a representable density
    // although its area Jacobian itself exceeds float range.
    float areaPdf = 0.0f;
};

struct AnalyticLightIntersection
{
    float distance = 0.0f;
    float3 point{};
    float3 normal{};
    float areaPdf = 0.0f;
    bool hit = false;
};

DEVICE_FUNC float analyticDiscArea(float3 axisX, float3 axisY)
{
    const float twiceParallelogramArea = finiteVectorLength(cross(axisX, axisY));
    return twiceParallelogramArea > 0.0f && twiceParallelogramArea <= 3.402823466e38f / M_PI_F ?
               M_PI_F * twiceParallelogramArea :
               0.0f;
}

DEVICE_FUNC float analyticDiscAreaPdf(float3 axisX, float3 axisY)
{
    const float areaPdf = finiteCrossReciprocal(axisX, axisY, 1.0f / M_PI_F);
    // Apple GPU safe math flushes subnormal arithmetic. Do not let the CPU
    // host assign selection mass to a conditional density the device cannot
    // represent under every supported Metal math mode.
    return areaPdf >= 1.175494351e-38f ? areaPdf : 0.0f;
}

DEVICE_FUNC float3 affineSphereCofactor(float3 axisX, float3 axisY, float3 axisZ, float3 objectNormal)
{
    return objectNormal.x * cross(axisY, axisZ) + objectNormal.y * cross(axisZ, axisX) +
           objectNormal.z * cross(axisX, axisY);
}

struct ScaledAffineBasis
{
    float3 x{};
    float3 y{};
    float3 z{};
    CompensatedFloat determinant{};
    int objectExponentX = 0;
    int objectExponentY = 0;
    int objectExponentZ = 0;
    int worldExponentX = 0;
    int worldExponentY = 0;
    int worldExponentZ = 0;
    bool valid = false;
};

DEVICE_FUNC bool affineVectorIsFinite(float3 value)
{
    return fabsf(value.x) <= 3.402823466e38f && fabsf(value.y) <= 3.402823466e38f &&
           fabsf(value.z) <= 3.402823466e38f;
}

DEVICE_FUNC bool affineSampleCoordinateRangeIsFinite(float center, float axisX, float axisY, float axisZ)
{
    constexpr float maxFinite = 3.402823466e38f;
    if (!(fabsf(center) <= maxFinite) || !(fabsf(axisX) <= maxFinite) || !(fabsf(axisY) <= maxFinite) ||
        !(fabsf(axisZ) <= maxFinite))
    {
        return false;
    }
    float remaining = maxFinite - fabsf(center);
    if (fabsf(axisX) > remaining)
    {
        return false;
    }
    remaining -= fabsf(axisX);
    if (fabsf(axisY) > remaining)
    {
        return false;
    }
    remaining -= fabsf(axisY);
    return fabsf(axisZ) <= remaining;
}

DEVICE_FUNC bool affineSamplePointRangeIsFinite(float3 center, float3 axisX, float3 axisY, float3 axisZ)
{
    return affineSampleCoordinateRangeIsFinite(center.x, axisX.x, axisY.x, axisZ.x) &&
           affineSampleCoordinateRangeIsFinite(center.y, axisX.y, axisY.y, axisZ.y) &&
           affineSampleCoordinateRangeIsFinite(center.z, axisX.z, axisY.z, axisZ.z);
}

DEVICE_FUNC bool affineComponentwiseInverseRowIsStable(float3 inverseNumeratorRow,
                                                       float3 x,
                                                       float3 y,
                                                       float3 z,
                                                       int rowObjectExponent,
                                                       int objectExponentX,
                                                       int objectExponentY,
                                                       int objectExponentZ,
                                                       float determinant)
{
    const float termX = fabsf(inverseNumeratorRow.x) * fabsf(x.x) +
                        fabsf(inverseNumeratorRow.y) * fabsf(x.y) +
                        fabsf(inverseNumeratorRow.z) * fabsf(x.z);
    const float termY = fabsf(inverseNumeratorRow.x) * fabsf(y.x) +
                        fabsf(inverseNumeratorRow.y) * fabsf(y.y) +
                        fabsf(inverseNumeratorRow.z) * fabsf(y.z);
    const float termZ = fabsf(inverseNumeratorRow.x) * fabsf(z.x) +
                        fabsf(inverseNumeratorRow.y) * fabsf(z.y) +
                        fabsf(inverseNumeratorRow.z) * fabsf(z.z);
    int exponentX = -100000;
    int exponentY = -100000;
    int exponentZ = -100000;
    float mantissaX = 0.0f;
    float mantissaY = 0.0f;
    float mantissaZ = 0.0f;
    if (termX > 0.0f)
    {
        mantissaX = decomposeFloatExponent(termX, exponentX);
        exponentX += objectExponentX - rowObjectExponent;
    }
    if (termY > 0.0f)
    {
        mantissaY = decomposeFloatExponent(termY, exponentY);
        exponentY += objectExponentY - rowObjectExponent;
    }
    if (termZ > 0.0f)
    {
        mantissaZ = decomposeFloatExponent(termZ, exponentZ);
        exponentZ += objectExponentZ - rowObjectExponent;
    }
    const int numeratorExponent = exponentX > exponentY ?
                                      (exponentX > exponentZ ? exponentX : exponentZ) :
                                      (exponentY > exponentZ ? exponentY : exponentZ);
    if (numeratorExponent == -100000)
    {
        return false;
    }
    const float numeratorMantissa = scaleFloatExponent(mantissaX, exponentX - numeratorExponent) +
                                    scaleFloatExponent(mantissaY, exponentY - numeratorExponent) +
                                    scaleFloatExponent(mantissaZ, exponentZ - numeratorExponent);
    int determinantExponent = 0;
    const float determinantMantissa = decomposeFloatExponent(fabsf(determinant), determinantExponent);
    const float condition = scaleFloatExponent(
        numeratorMantissa / determinantMantissa, numeratorExponent - determinantExponent);
    // A three-term float matrix-vector product has componentwise backward
    // error gamma_3 < 3 * 2^-24 / (1 - 3 * 2^-24). Keeping the Skeel
    // condition at 2^12 therefore bounds the recovered object-space endpoint
    // drift below 7.33e-4. General affine solves (including transformed
    // discs) do not map a unit sphere through A and do not inherit this gate.
    constexpr float maximumComponentwiseCondition = 4096.0f;
    return condition <= maximumComponentwiseCondition;
}

DEVICE_FUNC bool affineEllipsoidPointMapIsRepresentable(const THREAD_REF ScaledAffineBasis& basis)
{
    const float determinant = compensatedValue(basis.determinant);
    const float3 cofactorX = accurateCross(basis.y, basis.z);
    const float3 cofactorY = accurateCross(basis.z, basis.x);
    const float3 cofactorZ = accurateCross(basis.x, basis.y);
    return basis.valid &&
           affineComponentwiseInverseRowIsStable(cofactorX, basis.x, basis.y, basis.z, basis.objectExponentX,
                                                 basis.objectExponentX, basis.objectExponentY, basis.objectExponentZ,
                                                 determinant) &&
           affineComponentwiseInverseRowIsStable(cofactorY, basis.x, basis.y, basis.z, basis.objectExponentY,
                                                 basis.objectExponentX, basis.objectExponentY, basis.objectExponentZ,
                                                 determinant) &&
           affineComponentwiseInverseRowIsStable(cofactorZ, basis.x, basis.y, basis.z, basis.objectExponentZ,
                                                 basis.objectExponentX, basis.objectExponentY, basis.objectExponentZ,
                                                 determinant);
}

// Factor A=R*B*C using positive powers of two. C removes each authored
// axis's independent magnitude before R equilibrates the world rows. The
// resulting B retains the axis directions instead of rounding a small column
// away merely because another column is large.
DEVICE_FUNC ScaledAffineBasis scaledAffineBasis(float3 axisX, float3 axisY, float3 axisZ)
{
    ScaledAffineBasis result;
    if (!affineVectorIsFinite(axisX) || !affineVectorIsFinite(axisY) || !affineVectorIsFinite(axisZ))
    {
        return result;
    }

    const float scaleX = fmaxf(fabsf(axisX.x), fmaxf(fabsf(axisX.y), fabsf(axisX.z)));
    const float scaleY = fmaxf(fabsf(axisY.x), fmaxf(fabsf(axisY.y), fabsf(axisY.z)));
    const float scaleZ = fmaxf(fabsf(axisZ.x), fmaxf(fabsf(axisZ.y), fabsf(axisZ.z)));
    if (!(scaleX > 0.0f) || !(scaleY > 0.0f) || !(scaleZ > 0.0f))
    {
        return result;
    }

    int exponentX = 0;
    int exponentY = 0;
    int exponentZ = 0;
    decomposeFloatExponent(scaleX, exponentX);
    decomposeFloatExponent(scaleY, exponentY);
    decomposeFloatExponent(scaleZ, exponentZ);
    const int globalExponent = exponentX > exponentY ?
                                   (exponentX > exponentZ ? exponentX : exponentZ) :
                                   (exponentY > exponentZ ? exponentY : exponentZ);
    const float3 normalizedX = make_float3(scaleFloatExponent(axisX.x, -exponentX),
                                           scaleFloatExponent(axisX.y, -exponentX),
                                           scaleFloatExponent(axisX.z, -exponentX));
    const float3 normalizedY = make_float3(scaleFloatExponent(axisY.x, -exponentY),
                                           scaleFloatExponent(axisY.y, -exponentY),
                                           scaleFloatExponent(axisY.z, -exponentY));
    const float3 normalizedZ = make_float3(scaleFloatExponent(axisZ.x, -exponentZ),
                                           scaleFloatExponent(axisZ.y, -exponentZ),
                                           scaleFloatExponent(axisZ.z, -exponentZ));
    const float rowScaleX = fmaxf(fabsf(normalizedX.x), fmaxf(fabsf(normalizedY.x), fabsf(normalizedZ.x)));
    const float rowScaleY = fmaxf(fabsf(normalizedX.y), fmaxf(fabsf(normalizedY.y), fabsf(normalizedZ.y)));
    const float rowScaleZ = fmaxf(fabsf(normalizedX.z), fmaxf(fabsf(normalizedY.z), fabsf(normalizedZ.z)));
    if (!(rowScaleX > 0.0f) || !(rowScaleY > 0.0f) || !(rowScaleZ > 0.0f))
    {
        return result;
    }

    int rowExponentX = 0;
    int rowExponentY = 0;
    int rowExponentZ = 0;
    decomposeFloatExponent(rowScaleX, rowExponentX);
    decomposeFloatExponent(rowScaleY, rowExponentY);
    decomposeFloatExponent(rowScaleZ, rowExponentZ);
    result.x = make_float3(scaleFloatExponent(normalizedX.x, -rowExponentX),
                           scaleFloatExponent(normalizedX.y, -rowExponentY),
                           scaleFloatExponent(normalizedX.z, -rowExponentZ));
    result.y = make_float3(scaleFloatExponent(normalizedY.x, -rowExponentX),
                           scaleFloatExponent(normalizedY.y, -rowExponentY),
                           scaleFloatExponent(normalizedY.z, -rowExponentZ));
    result.z = make_float3(scaleFloatExponent(normalizedZ.x, -rowExponentX),
                           scaleFloatExponent(normalizedZ.y, -rowExponentY),
                           scaleFloatExponent(normalizedZ.z, -rowExponentZ));
    result.determinant = compensatedDotCrossExpansion(result.x, result.y, result.z);

    const float determinant = compensatedValue(result.determinant);
    const float3 cofactorX = accurateCross(result.y, result.z);
    const float3 cofactorY = accurateCross(result.z, result.x);
    const float3 cofactorZ = accurateCross(result.x, result.y);
    const float matrixNorm = fmaxf(fabsf(result.x.x) + fabsf(result.y.x) + fabsf(result.z.x),
                                   fmaxf(fabsf(result.x.y) + fabsf(result.y.y) + fabsf(result.z.y),
                                         fabsf(result.x.z) + fabsf(result.y.z) + fabsf(result.z.z)));
    const float adjugateNorm = fmaxf(fabsf(cofactorX.x) + fabsf(cofactorX.y) + fabsf(cofactorX.z),
                                     fmaxf(fabsf(cofactorY.x) + fabsf(cofactorY.y) + fabsf(cofactorY.z),
                                           fabsf(cofactorZ.x) + fabsf(cofactorZ.y) + fabsf(cofactorZ.z)));
    // With a two-float determinant, cond_inf(B)<=2^30 leaves at least about
    // eighteen meaningful result bits. Beyond that, the shared float inputs
    // cannot certify intersection support or orientation reliably.
    if (!(fabsf(determinant) > 9.313225746154785e-10f * matrixNorm * adjugateNorm))
    {
        return result;
    }

    result.objectExponentX = exponentX - globalExponent;
    result.objectExponentY = exponentY - globalExponent;
    result.objectExponentZ = exponentZ - globalExponent;
    result.worldExponentX = rowExponentX + globalExponent;
    result.worldExponentY = rowExponentY + globalExponent;
    result.worldExponentZ = rowExponentZ + globalExponent;
    result.valid = true;
    return result;
}

DEVICE_FUNC float analyticAffineOrientation(float3 axisX, float3 axisY, float3 axisZ)
{
    const ScaledAffineBasis basis = scaledAffineBasis(axisX, axisY, axisZ);
    const float determinant = compensatedValue(basis.determinant);
    return basis.valid ? (determinant > 0.0f ? 1.0f : -1.0f) : 0.0f;
}

DEVICE_FUNC int compensatedExponent(CompensatedFloat value, int externalExponent)
{
    const float magnitude = fmaxf(fabsf(value.high), fabsf(value.low));
    if (!(magnitude > 0.0f) || !(magnitude <= 3.402823466e38f))
    {
        return -100000;
    }
    int exponent = 0;
    decomposeFloatExponent(magnitude, exponent);
    return exponent + externalExponent;
}

DEVICE_FUNC float affineAxisScale(float3 axisX, float3 axisY, float3 axisZ)
{
    float scale = fmaxf(fabsf(axisX.x), fmaxf(fabsf(axisX.y), fabsf(axisX.z)));
    scale = fmaxf(scale, fmaxf(fabsf(axisY.x), fmaxf(fabsf(axisY.y), fabsf(axisY.z))));
    return fmaxf(scale, fmaxf(fabsf(axisZ.x), fmaxf(fabsf(axisZ.y), fabsf(axisZ.z))));
}

// Returns numerator / |cofactor(A) n| and its unit direction. Scaling each
// world row independently keeps every determinant term representable; keeping
// each component as a two-float expansion avoids the precision loss from
// materializing two nearly parallel transformed tangents before their cross.
DEVICE_FUNC float affineCofactorReciprocalAndDirection(float3 axisX,
                                                       float3 axisY,
                                                       float3 axisZ,
                                                       CompensatedFloat objectX,
                                                       CompensatedFloat objectY,
                                                       CompensatedFloat objectZ,
                                                       float numerator,
                                                       THREAD_REF float3& direction)
{
    direction = make_float3(0.0f);
    const float3 rowX = make_float3(axisX.x, axisY.x, axisZ.x);
    const float3 rowY = make_float3(axisX.y, axisY.y, axisZ.y);
    const float3 rowZ = make_float3(axisX.z, axisY.z, axisZ.z);
    const float scaleX = fmaxf(fabsf(rowX.x), fmaxf(fabsf(rowX.y), fabsf(rowX.z)));
    const float scaleY = fmaxf(fabsf(rowY.x), fmaxf(fabsf(rowY.y), fabsf(rowY.z)));
    const float scaleZ = fmaxf(fabsf(rowZ.x), fmaxf(fabsf(rowZ.y), fabsf(rowZ.z)));
    float inputScale = fmaxf(fabsf(objectX.high), fabsf(objectX.low));
    inputScale = fmaxf(inputScale, fmaxf(fabsf(objectY.high), fabsf(objectY.low)));
    inputScale = fmaxf(inputScale, fmaxf(fabsf(objectZ.high), fabsf(objectZ.low)));
    if (!(inputScale > 0.0f) || !(inputScale <= 3.402823466e38f) || !(scaleX > 0.0f) || !(scaleX <= 3.402823466e38f) ||
        !(scaleY > 0.0f) || !(scaleY <= 3.402823466e38f) || !(scaleZ > 0.0f) || !(scaleZ <= 3.402823466e38f))
    {
        return 0.0f;
    }

    int inputExponent = 0;
    int exponentX = 0;
    int exponentY = 0;
    int exponentZ = 0;
    decomposeFloatExponent(inputScale, inputExponent);
    decomposeFloatExponent(scaleX, exponentX);
    decomposeFloatExponent(scaleY, exponentY);
    decomposeFloatExponent(scaleZ, exponentZ);
    objectX = scaleCompensatedExponent(objectX, -inputExponent);
    objectY = scaleCompensatedExponent(objectY, -inputExponent);
    objectZ = scaleCompensatedExponent(objectZ, -inputExponent);
    const float3 x = make_float3(scaleFloatExponent(rowX.x, -exponentX), scaleFloatExponent(rowX.y, -exponentX),
                                 scaleFloatExponent(rowX.z, -exponentX));
    const float3 y = make_float3(scaleFloatExponent(rowY.x, -exponentY), scaleFloatExponent(rowY.y, -exponentY),
                                 scaleFloatExponent(rowY.z, -exponentY));
    const float3 z = make_float3(scaleFloatExponent(rowZ.x, -exponentZ), scaleFloatExponent(rowZ.y, -exponentZ),
                                 scaleFloatExponent(rowZ.z, -exponentZ));

    const CompensatedFloat inputLength =
        sqrtCompensated(compensatedDot3(objectX, objectY, objectZ, objectX, objectY, objectZ));
    const float inputLengthValue = compensatedValue(inputLength);
    const float cofactorX = compensatedValue(compensatedDotCrossExpansion(objectX, objectY, objectZ, y, z));
    const float cofactorY = compensatedValue(compensatedDotCrossExpansion(objectX, objectY, objectZ, z, x));
    const float cofactorZ = compensatedValue(compensatedDotCrossExpansion(objectX, objectY, objectZ, x, y));
    int componentExponentX = 0;
    int componentExponentY = 0;
    int componentExponentZ = 0;
    const float mantissaX = decomposeFloatExponent(cofactorX, componentExponentX);
    const float mantissaY = decomposeFloatExponent(cofactorY, componentExponentY);
    const float mantissaZ = decomposeFloatExponent(cofactorZ, componentExponentZ);
    const int totalExponentX = cofactorX != 0.0f ? exponentY + exponentZ + componentExponentX : -100000;
    const int totalExponentY = cofactorY != 0.0f ? exponentZ + exponentX + componentExponentY : -100000;
    const int totalExponentZ = cofactorZ != 0.0f ? exponentX + exponentY + componentExponentZ : -100000;
    const int commonExponent = totalExponentX > totalExponentY ?
                                   (totalExponentX > totalExponentZ ? totalExponentX : totalExponentZ) :
                                   (totalExponentY > totalExponentZ ? totalExponentY : totalExponentZ);
    if (commonExponent == -100000)
    {
        return 0.0f;
    }

    const float3 scaledCofactor = make_float3(scaleFloatExponent(mantissaX, totalExponentX - commonExponent),
                                              scaleFloatExponent(mantissaY, totalExponentY - commonExponent),
                                              scaleFloatExponent(mantissaZ, totalExponentZ - commonExponent));
    const float scaledLength = finiteVectorLength(scaledCofactor);
    direction = normalizeFiniteVectorOrZero(scaledCofactor);
    if (!(scaledLength > 0.0f) || !(dot(direction, direction) > 0.0f) || !(inputLengthValue > 0.0f) ||
        !(numerator > 0.0f) || !(numerator <= 3.402823466e38f))
    {
        return 0.0f;
    }

    int numeratorExponent = 0;
    int inputLengthExponent = 0;
    int lengthExponent = 0;
    const float numeratorMantissa = decomposeFloatExponent(numerator, numeratorExponent);
    const float inputLengthMantissa = decomposeFloatExponent(inputLengthValue, inputLengthExponent);
    const float lengthMantissa = decomposeFloatExponent(scaledLength, lengthExponent);
    const float reciprocal =
        scaleFloatExponent(numeratorMantissa * inputLengthMantissa / lengthMantissa,
                           numeratorExponent + inputLengthExponent - lengthExponent - commonExponent);
    return reciprocal >= 1.175494351e-38f && reciprocal <= 3.402823466e38f ? reciprocal : 0.0f;
}

DEVICE_FUNC float affineCofactorReciprocalAndDirection(
    float3 axisX, float3 axisY, float3 axisZ, float3 objectNormal, float numerator, THREAD_REF float3& direction)
{
    return affineCofactorReciprocalAndDirection(axisX, axisY, axisZ, compensatedSum(objectNormal.x, 0.0f),
                                                compensatedSum(objectNormal.y, 0.0f),
                                                compensatedSum(objectNormal.z, 0.0f), numerator, direction);
}

DEVICE_FUNC float affineSphereAreaPdfAndNormal(
    float3 axisX, float3 axisY, float3 axisZ, float3 objectNormal, THREAD_REF float3& normal)
{
    normal = make_float3(0.0f);
    const float orientation = analyticAffineOrientation(axisX, axisY, axisZ);
    float3 cofactorDirection;
    const float areaPdf = affineCofactorReciprocalAndDirection(
        axisX, axisY, axisZ, objectNormal, 1.0f / (4.0f * M_PI_F), cofactorDirection);
    normal = orientation * cofactorDirection;
    if (orientation == 0.0f || !(dot(normal, normal) > 0.0f) || !(areaPdf > 0.0f))
    {
        normal = make_float3(0.0f);
        return 0.0f;
    }
    return areaPdf;
}

// Unit inverse-transpose normal for an affine map whose columns are the three
// axes. The determinant sign matters for mirrored transforms; its magnitude
// cancels during normalization.
DEVICE_FUNC float3 transformAffineNormal(float3 axisX, float3 axisY, float3 axisZ, float3 objectNormal)
{
    const float orientation = analyticAffineOrientation(axisX, axisY, axisZ);
    float3 cofactorDirection;
    affineCofactorReciprocalAndDirection(axisX, axisY, axisZ, objectNormal, 1.0f, cofactorDirection);
    if (orientation == 0.0f || !(dot(cofactorDirection, cofactorDirection) > 0.0f))
    {
        return make_float3(0.0f);
    }
    return orientation * cofactorDirection;
}

DEVICE_FUNC bool solveAffineCoordinates(
    float3 axisX, float3 axisY, float3 axisZ, float3 worldOffset, THREAD_REF float3& objectCoordinates)
{
    const ScaledAffineBasis basis = scaledAffineBasis(axisX, axisY, axisZ);
    if (!basis.valid)
    {
        objectCoordinates = make_float3(0.0f);
        return false;
    }

    const float3 offset = make_float3(scaleFloatExponent(worldOffset.x, -basis.worldExponentX),
                                      scaleFloatExponent(worldOffset.y, -basis.worldExponentY),
                                      scaleFloatExponent(worldOffset.z, -basis.worldExponentZ));
    if (!affineVectorIsFinite(offset))
    {
        objectCoordinates = make_float3(0.0f);
        return false;
    }
    const float3 cofactorX = accurateCross(basis.y, basis.z);
    const float3 cofactorY = accurateCross(basis.z, basis.x);
    const float3 cofactorZ = accurateCross(basis.x, basis.y);
    float3 scaledCoordinates = make_float3(
        compensatedValue(divideCompensated(compensatedDotCrossExpansion(offset, basis.y, basis.z), basis.determinant)),
        compensatedValue(divideCompensated(compensatedDotCrossExpansion(offset, basis.z, basis.x), basis.determinant)),
        compensatedValue(divideCompensated(compensatedDotCrossExpansion(offset, basis.x, basis.y), basis.determinant)));
    if (!affineVectorIsFinite(scaledCoordinates))
    {
        objectCoordinates = make_float3(0.0f);
        return false;
    }
    // Cramer's rule supplies a scale-safe initial inverse. Two residual
    // corrections recover the low bits needed near a grazing intersection of
    // highly non-orthogonal, but still finite, affine axes.
    for (unsigned int iteration = 0u; iteration < 2u; ++iteration)
    {
        const float3 residual = make_float3(
            fmaf(-basis.z.x, scaledCoordinates.z,
                 fmaf(-basis.y.x, scaledCoordinates.y, fmaf(-basis.x.x, scaledCoordinates.x, offset.x))),
            fmaf(-basis.z.y, scaledCoordinates.z,
                 fmaf(-basis.y.y, scaledCoordinates.y, fmaf(-basis.x.y, scaledCoordinates.x, offset.y))),
            fmaf(-basis.z.z, scaledCoordinates.z,
                 fmaf(-basis.y.z, scaledCoordinates.y, fmaf(-basis.x.z, scaledCoordinates.x, offset.z))));
        const float3 correction = make_float3(
            compensatedValue(divideCompensated(compensatedDotExpansion(residual, cofactorX), basis.determinant)),
            compensatedValue(divideCompensated(compensatedDotExpansion(residual, cofactorY), basis.determinant)),
            compensatedValue(divideCompensated(compensatedDotExpansion(residual, cofactorZ), basis.determinant)));
        scaledCoordinates += correction;
    }
    objectCoordinates = make_float3(scaleFloatExponent(scaledCoordinates.x, -basis.objectExponentX),
                                    scaleFloatExponent(scaledCoordinates.y, -basis.objectExponentY),
                                    scaleFloatExponent(scaledCoordinates.z, -basis.objectExponentZ));
    return affineVectorIsFinite(objectCoordinates);
}

DEVICE_FUNC bool analyticAffineTransformIsNonsingular(float3 axisX, float3 axisY, float3 axisZ)
{
    const ScaledAffineBasis basis = scaledAffineBasis(axisX, axisY, axisZ);
    if (!affineEllipsoidPointMapIsRepresentable(basis))
    {
        return false;
    }
    // For a unit object normal n, J_A(n)=|cofactor(A)n|. Its maximum is at
    // most the sum of the three column norms. Express the reciprocal bound in
    // terms of representable basis densities, so no overflowing Jacobian is
    // formed and every point on an accepted ellipsoid has positive p_A.
    float3 nx;
    float3 ny;
    float3 nz;
    const float px = affineSphereAreaPdfAndNormal(axisX, axisY, axisZ, make_float3(1.0f, 0.0f, 0.0f), nx);
    const float py = affineSphereAreaPdfAndNormal(axisX, axisY, axisZ, make_float3(0.0f, 1.0f, 0.0f), ny);
    const float pz = affineSphereAreaPdfAndNormal(axisX, axisY, axisZ, make_float3(0.0f, 0.0f, 1.0f), nz);
    const float minimumPdf = fminf(px, fminf(py, pz));
    if (!(minimumPdf > 0.0f))
    {
        return false;
    }
    // Bound the largest cofactor singular value with the infinity norm of its
    // normalized Gram matrix. Unlike a sum of column lengths, this is exact
    // for orthogonal/isotropic axes at the last positive float PDF.
    const float rx = minimumPdf / px;
    const float ry = minimumPdf / py;
    const float rz = minimumPdf / pz;
    const float xy = rx * ry * fminf(fabsf(dot(nx, ny)), 1.0f);
    const float xz = rx * rz * fminf(fabsf(dot(nx, nz)), 1.0f);
    const float yz = ry * rz * fminf(fabsf(dot(ny, nz)), 1.0f);
    const float gramNorm = fmaxf(rx * rx + xy + xz, fmaxf(ry * ry + xy + yz, rz * rz + xz + yz));
    const float conservativePdf = minimumPdf / sqrtf(gramNorm);
    return conservativePdf > 0.0f;
}

DEVICE_FUNC bool analyticEllipsoidIsRepresentable(float3 center, float3 axisX, float3 axisY, float3 axisZ)
{
    return affineSamplePointRangeIsFinite(center, axisX, axisY, axisZ) &&
           analyticAffineTransformIsNonsingular(axisX, axisY, axisZ);
}

DEVICE_FUNC float3 affineSphereCoordinates(float3 axisX, float3 axisY, float3 axisZ, float3 worldOffset)
{
    const float determinant = dot(axisX, cross(axisY, axisZ));
    if (!(fabsf(determinant) > 0.0f))
    {
        return make_float3(0.0f);
    }
    return make_float3(dot(worldOffset, cross(axisY, axisZ)), dot(worldOffset, cross(axisZ, axisX)),
                       dot(worldOffset, cross(axisX, axisY))) /
           determinant;
}

DEVICE_FUNC AnalyticLightSample
sampleAnalyticDisc(float3 center, float3 axisX, float3 axisY, float3 emissionNormal, float u1, float u2)
{
    AnalyticLightSample sample;
    const float areaPdf = analyticDiscAreaPdf(axisX, axisY);
    if (!affineSamplePointRangeIsFinite(center, axisX, axisY, make_float3(0.0f)) || !(areaPdf > 0.0f) ||
        !(dot(emissionNormal, emissionNormal) > 0.0f))
    {
        return sample;
    }
    const float radius = sqrtf(fminf(fmaxf(u1, 0.0f), 1.0f));
    const float phi = 2.0f * M_PI_F * u2;
    sample.point = center + radius * (cosf(phi) * axisX + sinf(phi) * axisY);
    sample.normal = emissionNormal;
    sample.areaPdf = areaPdf;
    return sample;
}

DEVICE_FUNC AnalyticLightSample
sampleAnalyticEllipsoid(float3 center, float3 axisX, float3 axisY, float3 axisZ, float u1, float u2)
{
    AnalyticLightSample sample;
    if (!analyticEllipsoidIsRepresentable(center, axisX, axisY, axisZ))
    {
        return sample;
    }
    const float z = 1.0f - 2.0f * u1;
    const float radial = sqrtf(fmaxf(1.0f - z * z, 0.0f));
    const float phi = 2.0f * M_PI_F * u2;
    float3 objectNormal = make_float3(radial * cosf(phi), radial * sinf(phi), z);
    const float objectLengthSquared = dot(objectNormal, objectNormal);
    if (objectLengthSquared > 0.0f)
    {
        objectNormal /= sqrtf(objectLengthSquared);
    }
    sample.point = make_float3(
        fmaf(objectNormal.z, axisZ.x, fmaf(objectNormal.y, axisY.x, fmaf(objectNormal.x, axisX.x, center.x))),
        fmaf(objectNormal.z, axisZ.y, fmaf(objectNormal.y, axisY.y, fmaf(objectNormal.x, axisX.y, center.y))),
        fmaf(objectNormal.z, axisZ.z, fmaf(objectNormal.y, axisY.z, fmaf(objectNormal.x, axisX.z, center.z))));
    sample.areaPdf = affineSphereAreaPdfAndNormal(axisX, axisY, axisZ, objectNormal, sample.normal);
    return sample;
}

DEVICE_FUNC float analyticEllipsoidAreaPdf(
    float3 center, float3 axisX, float3 axisY, float3 axisZ, float3 point, THREAD_REF float3& normal)
{
    if (!analyticEllipsoidIsRepresentable(center, axisX, axisY, axisZ))
    {
        normal = make_float3(0.0f);
        return 0.0f;
    }
    float3 objectNormal;
    if (!solveAffineCoordinates(axisX, axisY, axisZ, point - center, objectNormal))
    {
        normal = make_float3(0.0f);
        return 0.0f;
    }
    objectNormal = normalizeFiniteVectorOrZero(objectNormal);
    if (!(dot(objectNormal, objectNormal) > 0.0f))
    {
        normal = make_float3(0.0f);
        return 0.0f;
    }
    return affineSphereAreaPdfAndNormal(axisX, axisY, axisZ, objectNormal, normal);
}

// Deterministic equal-area quadrature of the smooth ellipsoid's surface area.
// Sampling/PDF evaluation does not depend on this approximation; it is used
// only to power-weight the outer light-selection PMF.
DEVICE_FUNC float analyticEllipsoidSurfaceArea(float3 axisX, float3 axisY, float3 axisZ)
{
    if (!analyticAffineTransformIsNonsingular(axisX, axisY, axisZ))
    {
        return 0.0f;
    }
    const unsigned int sampleCount = 256u;
    const float goldenAngle = 2.39996322972865332f;
    float jacobianMean = 0.0f;
    for (unsigned int i = 0u; i < sampleCount; ++i)
    {
        const float z = 1.0f - 2.0f * (float(i) + 0.5f) / float(sampleCount);
        const float radial = sqrtf(fmaxf(1.0f - z * z, 0.0f));
        const float phi = goldenAngle * float(i);
        const float3 objectNormal = make_float3(radial * cosf(phi), radial * sinf(phi), z);
        float3 normal;
        const float areaPdf = affineSphereAreaPdfAndNormal(axisX, axisY, axisZ, objectNormal, normal);
        if (!(areaPdf > 0.0f))
        {
            return 0.0f;
        }
        const float jacobian = 1.0f / (4.0f * M_PI_F * areaPdf);
        if (!(jacobian > 0.0f) || !(jacobian <= 3.402823466e38f))
        {
            return 0.0f;
        }
        jacobianMean += jacobian / float(sampleCount);
    }
    return 4.0f * M_PI_F * jacobianMean;
}

DEVICE_FUNC AnalyticLightIntersection intersectAnalyticDisc(float3 rayOrigin,
                                                            float3 rayDirection,
                                                            float minDistance,
                                                            float maxDistance,
                                                            float3 center,
                                                            float3 axisX,
                                                            float3 axisY,
                                                            float3 emissionNormal)
{
    AnalyticLightIntersection result;
    result.distance = maxDistance;
    result.point = make_float3(0.0f);
    result.normal = make_float3(0.0f);
    result.areaPdf = 0.0f;
    result.hit = false;

    const float3 planeNormal = finiteCrossDirection(axisX, axisY);
    const float areaPdf = analyticDiscAreaPdf(axisX, axisY);
    if (!affineSamplePointRangeIsFinite(center, axisX, axisY, make_float3(0.0f)) ||
        !(dot(planeNormal, planeNormal) > 0.0f) || !(dot(emissionNormal, emissionNormal) > 0.0f) || !(areaPdf > 0.0f))
    {
        return result;
    }
    const float denominator = accurateDot(rayDirection, planeNormal);
    if (!(fabsf(denominator) > 0.0f))
    {
        return result;
    }
    const float numerator = accurateDot(center - rayOrigin, planeNormal);
    const float distanceHigh = numerator / denominator;
    const float distanceLow = fmaf(-distanceHigh, denominator, numerator) / denominator;
    const float distance = distanceHigh + distanceLow;
    if (!(distance >= minDistance && distance < maxDistance))
    {
        return result;
    }

    // Retain the quotient remainder while reconstructing a far hit. A single
    // float distance can lose an entire small disc even though the ray and its
    // local intersection remain representable.
    const float3 relativeOrigin = rayOrigin - center;
    const float3 offset = make_float3(fmaf(distanceHigh, rayDirection.x, relativeOrigin.x),
                                      fmaf(distanceHigh, rayDirection.y, relativeOrigin.y),
                                      fmaf(distanceHigh, rayDirection.z, relativeOrigin.z)) +
                          distanceLow * rayDirection;
    float3 coordinates;
    if (!solveAffineCoordinates(axisX, axisY, planeNormal, offset, coordinates) ||
        !(coordinates.x * coordinates.x + coordinates.y * coordinates.y <= 1.0f))
    {
        return result;
    }

    result.distance = distance;
    result.point = center + coordinates.x * axisX + coordinates.y * axisY;
    result.normal = emissionNormal;
    result.areaPdf = areaPdf;
    result.hit = true;
    return result;
}

DEVICE_FUNC bool affineSphereHitCoordinateHasSmallResidual(float center,
                                                           float axisX,
                                                           float axisY,
                                                           float axisZ,
                                                           float objectX,
                                                           float objectY,
                                                           float objectZ,
                                                           float rayOrigin,
                                                           float rayDirection,
                                                           CompensatedFloat distance)
{
    const float surfaceX = axisX * objectX;
    const float surfaceY = axisY * objectY;
    const float surfaceZ = axisZ * objectZ;
    const float rayHigh = rayDirection * distance.high;
    const float rayLow = rayDirection * distance.low;
    float scale = fmaxf(fabsf(center), fabsf(rayOrigin));
    scale = fmaxf(scale, fmaxf(fabsf(surfaceX), fmaxf(fabsf(surfaceY), fabsf(surfaceZ))));
    scale = fmaxf(scale, fmaxf(fabsf(rayHigh), fabsf(rayLow)));
    if (!(scale <= 3.402823466e38f))
    {
        return false;
    }
    if (!(scale > 0.0f))
    {
        return true;
    }

    int exponent = 0;
    decomposeFloatExponent(scale, exponent);
    const float scaledCenter = scaleFloatExponent(center, -exponent);
    const float scaledOrigin = scaleFloatExponent(rayOrigin, -exponent);
    const float scaledSurfaceX = scaleFloatExponent(surfaceX, -exponent);
    const float scaledSurfaceY = scaleFloatExponent(surfaceY, -exponent);
    const float scaledSurfaceZ = scaleFloatExponent(surfaceZ, -exponent);
    const float scaledRayHigh = scaleFloatExponent(rayHigh, -exponent);
    const float scaledRayLow = scaleFloatExponent(rayLow, -exponent);
    CompensatedFloat residual = compensatedSum(scaledCenter, scaledSurfaceX);
    residual = addCompensated(residual, compensatedSum(scaledSurfaceY, scaledSurfaceZ));
    residual = addCompensated(residual, compensatedSum(-scaledOrigin, -scaledRayHigh));
    residual = addCompensated(residual, compensatedSum(-scaledRayLow, 0.0f));
    const float absoluteSum = fabsf(scaledCenter) + fabsf(scaledSurfaceX) + fabsf(scaledSurfaceY) +
                              fabsf(scaledSurfaceZ) + fabsf(scaledOrigin) + fabsf(scaledRayHigh) +
                              fabsf(scaledRayLow);
    constexpr float residualTolerance = 1.9073486328125e-6f; // 32 * 2^-24
    return fabsf(compensatedValue(residual)) <= residualTolerance * absoluteSum;
}

DEVICE_FUNC bool affineSphereHitHasSmallResidual(float3 center,
                                                 float3 axisX,
                                                 float3 axisY,
                                                 float3 axisZ,
                                                 float3 objectNormal,
                                                 float3 rayOrigin,
                                                 float3 rayDirection,
                                                 CompensatedFloat distance)
{
    return affineSphereHitCoordinateHasSmallResidual(center.x, axisX.x, axisY.x, axisZ.x, objectNormal.x,
                                                     objectNormal.y, objectNormal.z, rayOrigin.x, rayDirection.x,
                                                     distance) &&
           affineSphereHitCoordinateHasSmallResidual(center.y, axisX.y, axisY.y, axisZ.y, objectNormal.x,
                                                     objectNormal.y, objectNormal.z, rayOrigin.y, rayDirection.y,
                                                     distance) &&
           affineSphereHitCoordinateHasSmallResidual(center.z, axisX.z, axisY.z, axisZ.z, objectNormal.x,
                                                     objectNormal.y, objectNormal.z, rayOrigin.z, rayDirection.z,
                                                     distance);
}

DEVICE_FUNC AnalyticLightIntersection intersectAnalyticEllipsoid(float3 rayOrigin,
                                                                 float3 rayDirection,
                                                                 float minDistance,
                                                                 float maxDistance,
                                                                 float3 center,
                                                                 float3 axisX,
                                                                 float3 axisY,
                                                                 float3 axisZ)
{
    AnalyticLightIntersection result;
    result.distance = maxDistance;
    result.point = make_float3(0.0f);
    result.normal = make_float3(0.0f);
    result.areaPdf = 0.0f;
    result.hit = false;

    if (!analyticEllipsoidIsRepresentable(center, axisX, axisY, axisZ))
    {
        return result;
    }
    const ScaledAffineBasis basis = scaledAffineBasis(axisX, axisY, axisZ);
    if (!basis.valid)
    {
        return result;
    }
    const float3 relativeOrigin = rayOrigin - center;
    const CompensatedFloat directionLengthSquaredExpansion = compensatedDotExpansion(rayDirection, rayDirection);
    const float directionLengthSquared = compensatedValue(directionLengthSquaredExpansion);
    if (!(directionLengthSquared > 0.0f) || !(directionLengthSquared <= 3.402823466e38f))
    {
        return result;
    }
    const CompensatedFloat shift = divideCompensated(
        negateCompensated(compensatedDotExpansion(relativeOrigin, rayDirection)), directionLengthSquaredExpansion);
    const float shiftHigh = shift.high;
    const float shiftLow = shift.low;
    const float3 scaledOrigin = make_float3(scaleFloatExponent(relativeOrigin.x, -basis.worldExponentX),
                                            scaleFloatExponent(relativeOrigin.y, -basis.worldExponentY),
                                            scaleFloatExponent(relativeOrigin.z, -basis.worldExponentZ));
    const float3 scaledDirection =
        make_float3(scaleFloatExponent(rayDirection.x, -basis.worldExponentX),
                                               scaleFloatExponent(rayDirection.y, -basis.worldExponentY),
                                               scaleFloatExponent(rayDirection.z, -basis.worldExponentZ));
    if (!affineVectorIsFinite(scaledOrigin) || !affineVectorIsFinite(scaledDirection))
    {
        return result;
    }
    CompensatedFloat determinant = basis.determinant;
    const CompensatedFloat directionX = compensatedDotCrossExpansion(scaledDirection, basis.y, basis.z);
    const CompensatedFloat directionY = compensatedDotCrossExpansion(scaledDirection, basis.z, basis.x);
    const CompensatedFloat directionZ = compensatedDotCrossExpansion(scaledDirection, basis.x, basis.y);
    const CompensatedFloat shiftedWorldX =
        addCompensated(addCompensated(compensatedSum(scaledOrigin.x, 0.0f),
                                      scaleCompensated(compensatedSum(scaledDirection.x, 0.0f), shiftHigh)),
                       scaleCompensated(compensatedSum(scaledDirection.x, 0.0f), shiftLow));
    const CompensatedFloat shiftedWorldY =
        addCompensated(addCompensated(compensatedSum(scaledOrigin.y, 0.0f),
                                      scaleCompensated(compensatedSum(scaledDirection.y, 0.0f), shiftHigh)),
                       scaleCompensated(compensatedSum(scaledDirection.y, 0.0f), shiftLow));
    const CompensatedFloat shiftedWorldZ =
        addCompensated(addCompensated(compensatedSum(scaledOrigin.z, 0.0f),
                                      scaleCompensated(compensatedSum(scaledDirection.z, 0.0f), shiftHigh)),
                       scaleCompensated(compensatedSum(scaledDirection.z, 0.0f), shiftLow));
    const CompensatedFloat originX =
        compensatedDotCrossExpansion(shiftedWorldX, shiftedWorldY, shiftedWorldZ, basis.y, basis.z);
    const CompensatedFloat originY =
        compensatedDotCrossExpansion(shiftedWorldX, shiftedWorldY, shiftedWorldZ, basis.z, basis.x);
    const CompensatedFloat originZ =
        compensatedDotCrossExpansion(shiftedWorldX, shiftedWorldY, shiftedWorldZ, basis.x, basis.y);
    const int originComponentExponentX = compensatedExponent(originX, -basis.objectExponentX);
    const int originComponentExponentY = compensatedExponent(originY, -basis.objectExponentY);
    const int originComponentExponentZ = compensatedExponent(originZ, -basis.objectExponentZ);
    const int determinantExponent = compensatedExponent(determinant, 0);
    const int directionComponentExponentX = compensatedExponent(directionX, -basis.objectExponentX);
    const int directionComponentExponentY = compensatedExponent(directionY, -basis.objectExponentY);
    const int directionComponentExponentZ = compensatedExponent(directionZ, -basis.objectExponentZ);
    const int originExponent = originComponentExponentX > originComponentExponentY ?
                                   (originComponentExponentX > originComponentExponentZ ?
                                        originComponentExponentX :
                                        originComponentExponentZ) :
                                   (originComponentExponentY > originComponentExponentZ ?
                                        originComponentExponentY :
                                        originComponentExponentZ);
    const int completeOriginExponent = originExponent > determinantExponent ? originExponent : determinantExponent;
    const int directionExponent = directionComponentExponentX > directionComponentExponentY ?
                                      (directionComponentExponentX > directionComponentExponentZ ?
                                           directionComponentExponentX :
                                           directionComponentExponentZ) :
                                      (directionComponentExponentY > directionComponentExponentZ ?
                                           directionComponentExponentY :
                                           directionComponentExponentZ);
    if (completeOriginExponent == -100000 || directionExponent == -100000)
    {
        return result;
    }
    const CompensatedFloat sx =
        scaleCompensatedExponent(originX, -basis.objectExponentX - completeOriginExponent);
    const CompensatedFloat sy =
        scaleCompensatedExponent(originY, -basis.objectExponentY - completeOriginExponent);
    const CompensatedFloat sz =
        scaleCompensatedExponent(originZ, -basis.objectExponentZ - completeOriginExponent);
    const CompensatedFloat dx =
        scaleCompensatedExponent(directionX, -basis.objectExponentX - directionExponent);
    const CompensatedFloat dy =
        scaleCompensatedExponent(directionY, -basis.objectExponentY - directionExponent);
    const CompensatedFloat dz =
        scaleCompensatedExponent(directionZ, -basis.objectExponentZ - directionExponent);
    determinant = scaleCompensatedExponent(determinant, -completeOriginExponent);

    const CompensatedFloat a = compensatedDot3(dx, dy, dz, dx, dy, dz);
    const CompensatedFloat halfB = compensatedDot3(sx, sy, sz, dx, dy, dz);
    // The usual b^2-a*c form subtracts two values set by the distance to the
    // origin; for a small affine light far away it can need more significand
    // bits than even a two-float expansion contains. The equivalent geometric
    // discriminant det(B)^2 |D|^2 - |O cross D|^2 depends only on the line's
    // perpendicular distance and retains the scale of the light itself.
    const CompensatedFloat crossX =
        addCompensated(multiplyCompensated(sy, dz), negateCompensated(multiplyCompensated(sz, dy)));
    const CompensatedFloat crossY =
        addCompensated(multiplyCompensated(sz, dx), negateCompensated(multiplyCompensated(sx, dz)));
    const CompensatedFloat crossZ =
        addCompensated(multiplyCompensated(sx, dy), negateCompensated(multiplyCompensated(sy, dx)));
    const CompensatedFloat perpendicularSquared =
        compensatedDot3(crossX, crossY, crossZ, crossX, crossY, crossZ);
    const CompensatedFloat discriminant =
        addCompensated(multiplyCompensated(multiplyCompensated(determinant, determinant), a),
                       negateCompensated(perpendicularSquared));
    const float aValue = compensatedValue(a);
    const float discriminantValue = compensatedValue(discriminant);
    const float determinantValue = compensatedValue(determinant);
    if (!(aValue > 0.0f) || !(discriminantValue >= 0.0f) || !(fabsf(determinantValue) > 0.0f))
    {
        return result;
    }
    const CompensatedFloat closestParameter = divideCompensated(negateCompensated(halfB), a);
    const CompensatedFloat root = divideCompensated(sqrtCompensated(discriminant), a);
    // Reconstruct the surface point from the line's perpendicular component
    // and chord length. Evaluating s+d*(closest+-root) loses the chord when a
    // very thin ellipsoid is far from the ray origin, even though both roots
    // remain representable as a two-float world distance.
    const CompensatedFloat perpendicularX = divideCompensated(
        addCompensated(multiplyCompensated(dy, crossZ), negateCompensated(multiplyCompensated(dz, crossY))),
        a);
    const CompensatedFloat perpendicularY = divideCompensated(
        addCompensated(multiplyCompensated(dz, crossX), negateCompensated(multiplyCompensated(dx, crossZ))),
        a);
    const CompensatedFloat perpendicularZ = divideCompensated(
        addCompensated(multiplyCompensated(dx, crossY), negateCompensated(multiplyCompensated(dy, crossX))),
        a);
    CompensatedFloat pointX =
        addCompensated(perpendicularX, negateCompensated(multiplyCompensated(dx, root)));
    CompensatedFloat pointY =
        addCompensated(perpendicularY, negateCompensated(multiplyCompensated(dy, root)));
    CompensatedFloat pointZ =
        addCompensated(perpendicularZ, negateCompensated(multiplyCompensated(dz, root)));
    const CompensatedFloat worldClosest =
        scaleCompensatedExponent(closestParameter, completeOriginExponent - directionExponent);
    const CompensatedFloat worldRoot = scaleCompensatedExponent(root, completeOriginExponent - directionExponent);
    const CompensatedFloat shiftedWorldClosest = addCompensated(compensatedSum(shiftHigh, shiftLow), worldClosest);
    CompensatedFloat distanceExpansion = addCompensated(shiftedWorldClosest, negateCompensated(worldRoot));
    float distance = compensatedValue(distanceExpansion);
    if (!(distance >= minDistance && distance < maxDistance))
    {
        pointX = addCompensated(perpendicularX, multiplyCompensated(dx, root));
        pointY = addCompensated(perpendicularY, multiplyCompensated(dy, root));
        pointZ = addCompensated(perpendicularZ, multiplyCompensated(dz, root));
        distanceExpansion = addCompensated(shiftedWorldClosest, worldRoot);
        distance = compensatedValue(distanceExpansion);
    }
    if (!(distance >= minDistance && distance < maxDistance))
    {
        return result;
    }

    const float3 homogeneousPoint =
        make_float3(compensatedValue(pointX), compensatedValue(pointY), compensatedValue(pointZ));
    const float3 objectNormal = copysignf(1.0f, determinantValue) * normalizeFiniteVectorOrZero(homogeneousPoint);
    if (!affineSphereHitHasSmallResidual(center, axisX, axisY, axisZ, objectNormal, rayOrigin, rayDirection, distanceExpansion))
    {
        return result;
    }
    float3 cofactorDirection;
    const float areaPdf = affineCofactorReciprocalAndDirection(
        axisX, axisY, axisZ, pointX, pointY, pointZ, 1.0f / (4.0f * M_PI_F), cofactorDirection);
    const float normalSign = analyticAffineOrientation(axisX, axisY, axisZ) * copysignf(1.0f, determinantValue);
    const float3 normal = normalSign * cofactorDirection;
    if (!(areaPdf > 0.0f) || !(dot(normal, normal) > 0.0f))
    {
        return result;
    }
    result.distance = distance;
    result.point = center + objectNormal.x * axisX + objectNormal.y * axisY + objectNormal.z * axisZ;
    result.normal = normal;
    result.areaPdf = areaPdf;
    result.hit = true;
    return result;
}

#endif // STRELKA_ANALYTIC_LIGHT_H
