#ifndef STRELKA_IES_MATH_H
#define STRELKA_IES_MATH_H

// ============================================================================
// ies_math.h -- evaluating an IES photometric table.
//
// One implementation for the CPU loader and both device backends. It used to be
// three transcriptions, and all three carried the same mistakes -- which is what
// three copies of a convention buys.
//
// The interpolation is Catmull-Rom over four samples per axis, not bilinear,
// and that is a deliberate match to Cycles (intern/cycles/kernel/util/ies.h).
// The reason is measurable rather than aesthetic: on the ladder's Philips
// CDM-R111 reflector, whose table steps 5 degrees while the beam falls by a
// factor of 2.4 between 20 and 25 degrees, straight lines between the samples
// read 12.7% high at the midpoint of that interval. Against the Cycles
// reference the row sat at ratio 1.050 with rel 0.057; the whole of that gap
// was the two renderers drawing different curves through the same numbers.
//
// A table is also the *whole* of what the luminaire emits: outside its
// tabulated range the answer is zero, not the edge value extrapolated onwards
// (which reached -1800 cd at 180 degrees on a downlight, and +2800 on a table
// that rises to its last entry). Cycles returns zero there too.
//
// The angles are expected already unfolded to the full turn -- see
// unfoldIesAzimuth() in the loader, which mirrors a quadrant or half table the
// way LM-63 says to and appends the 360 degree duplicate. Doing it once at load
// time is what lets the cubic have real neighbours at the seam instead of
// reflected guesses.
// ============================================================================

#include <strelka/material/material_math.h>

// Which address space the table lives in. Metal needs it spelled out on the
// pointer; the other two compilers have one address space and want nothing.
#if defined(__METAL_VERSION__)
#    define STRELKA_IES_PTR device
#else
#    define STRELKA_IES_PTR
#endif

/// Catmull-Rom through four samples, evaluated at x in [0, 1] between b and c.
/// Same expression as Cycles' cubic_interp(), so the two agree sample for
/// sample and not merely in shape.
DEVICE_FUNC float iesCubicInterp(float a, float b, float c, float d, float x)
{
    return 0.5f * (((d + 3.0f * (b - c) - a) * x + (2.0f * a - 5.0f * b + 4.0f * c - d)) * x + (c - a)) * x + b;
}

/// True when two angles are the same to within a hair, for the wrap tests.
DEVICE_FUNC bool iesAngleClose(float a, float b)
{
    return fabsf(a - b) < 1e-4f;
}

/// The index of the interval containing `x` in the ascending table `a`, clamped
/// so both it and it+1 are addressable.
DEVICE_FUNC int iesLowerIndex(STRELKA_IES_PTR const float* a, int n, float x)
{
    int lo = 0;
    int hi = n;
    while (lo < hi)
    {
        const int mid = (lo + hi) / 2;
        if (a[mid] < x)
        {
            lo = mid + 1;
        }
        else
        {
            hi = mid;
        }
    }
    const int idx = lo - 1;
    return (idx < 0) ? 0 : ((idx > n - 2) ? (n - 2) : idx);
}

/// One column of the table, interpolated across the vertical axis.
///
/// The fallbacks at the ends are Cycles': a missing first neighbour repeats the
/// second, unless the table starts at the pole, where the value across the pole
/// at this azimuth is a better guess than a flat repeat. Same at the far end.
DEVICE_FUNC float iesInterpVertical(
    STRELKA_IES_PTR const float* candela, int nV, int h, int v, float vFrac, bool wrapLow, bool wrapHigh)
{
    const float c = candela[(v + 1) + h * nV];
    const float b = candela[v + h * nV];

    float a = b;
    if (v > 0)
    {
        a = candela[(v - 1) + h * nV];
    }
    else if (wrapLow)
    {
        a = candela[1 + h * nV];
    }

    float d = c;
    if (v + 2 < nV)
    {
        d = candela[(v + 2) + h * nV];
    }
    else if (wrapHigh)
    {
        d = candela[(nV - 2) + h * nV];
    }

    return iesCubicInterp(a, b, c, d, vFrac);
}

/// Candela in the direction (vertDeg from the photometric axis, azimuthDeg
/// around it), for a table already unfolded to the full azimuth range.
///
/// Returns zero outside the tabulated range: the file describes everything the
/// luminaire emits, so a direction it does not cover receives nothing.
DEVICE_FUNC float iesEvaluate(STRELKA_IES_PTR const float* vAngles,
                              int nV,
                              STRELKA_IES_PTR const float* hAngles,
                              int nH,
                              STRELKA_IES_PTR const float* candela,
                              float vertDeg,
                              float azimuthDeg)
{
    if (nV < 2 || nH < 2)
    {
        return 0.0f;
    }

    float h = fmodf(azimuthDeg, 360.0f);
    if (h < 0.0f)
    {
        h += 360.0f;
    }

    const float vLow = vAngles[0];
    const float vHigh = vAngles[nV - 1];
    const float hLow = hAngles[0];
    const float hHigh = hAngles[nH - 1];
    if (vertDeg < vLow || vertDeg > vHigh || h < hLow || h > hHigh)
    {
        return 0.0f;
    }

    // A table spanning the whole turn, or reaching a pole, has real neighbours
    // across the seam; one that stops short does not, and the cubic falls back
    // to repeating an endpoint there.
    const bool wrapH = iesAngleClose(hLow, 0.0f) && iesAngleClose(hHigh, 360.0f);
    const bool wrapVLow = iesAngleClose(vLow, 0.0f);
    const bool wrapVHigh = iesAngleClose(vHigh, 180.0f);

    const int vi = iesLowerIndex(vAngles, nV, vertDeg);
    const int hi = iesLowerIndex(hAngles, nH, h);

    const float vSpan = vAngles[vi + 1] - vAngles[vi];
    const float hSpan = hAngles[hi + 1] - hAngles[hi];
    const float vFrac = (vSpan > 0.0f) ? fminf(fmaxf((vertDeg - vAngles[vi]) / vSpan, 0.0f), 1.0f) : 0.0f;
    const float hFrac = (hSpan > 0.0f) ? fminf(fmaxf((h - hAngles[hi]) / hSpan, 0.0f), 1.0f) : 0.0f;

    const float b = iesInterpVertical(candela, nV, hi, vi, vFrac, wrapVLow, wrapVHigh);
    const float c = iesInterpVertical(candela, nV, hi + 1, vi, vFrac, wrapVLow, wrapVHigh);

    float a = b;
    if (hi > 0)
    {
        a = iesInterpVertical(candela, nV, hi - 1, vi, vFrac, wrapVLow, wrapVHigh);
    }
    else if (wrapH)
    {
        // The last column (360 degrees) repeats the first, so the neighbour
        // before the seam is the one before that.
        a = iesInterpVertical(candela, nV, nH - 2, vi, vFrac, wrapVLow, wrapVHigh);
    }

    float d = b;
    if (hi + 2 < nH)
    {
        d = iesInterpVertical(candela, nV, hi + 2, vi, vFrac, wrapVLow, wrapVHigh);
    }
    else if (wrapH)
    {
        d = iesInterpVertical(candela, nV, 1, vi, vFrac, wrapVLow, wrapVHigh);
    }

    // A cubic through non-negative samples can still dip below zero on a steep
    // flank, and a negative candela is a light that subtracts.
    return fmaxf(iesCubicInterp(a, b, c, d, hFrac), 0.0f);
}

#endif // STRELKA_IES_MATH_H
