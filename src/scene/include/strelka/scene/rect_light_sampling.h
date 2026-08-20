#pragma once

// Host-side rectangular light sampling: area-uniform and solid-angle
// (Ureña / Fajardo / King, EGSR 2013 — "An Area-Preserving Parametrization for
// Spherical Rectangles"). Mirrors the Metal/CUDA shader paths so the unit tests
// exercise the same maths the GPU runs.
//
// The solid-angle path follows Cycles' numerically hardened form of that paper:
// internal angles via asin instead of acos (avoids cancellation when the solid
// angle is a tiny leftover from ~2π), and the sampling constants rewritten so
// the cancelled +π never appears. See blender/cycles src/kernel/light/area.h.

#include <algorithm>
#include <cmath>

#include <strelka/scene/glm_wrapper.hpp>

// The spherical rectangle itself. This header used to carry its own
// transcription of it -- and tests/scene/test_rect_light_sampling.cpp, the only
// test of the construction in the tree, therefore covered a copy that no GPU
// ever runs. Both shader backends now compile the same file.
#include <rect_sampling.h>


namespace oka::rect_light_sampling
{

struct RectCorners
{
    glm::float3 p0{};
    glm::float3 p1{};
    glm::float3 p3{};
};

struct LightSample
{
    glm::float3 pointOnLight{};
    glm::float3 L{};
    glm::float3 normal{};
    float distToLight = 0.0f;
    float area = 0.0f;
    float pdf = 0.0f;
};

// The shared construction, in this namespace and taking the corner triple this
// header works with.
using ::SphQuad;

inline float length3(const glm::float3& v)
{
    return std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
}

inline RectCorners fromWidthHeight(const glm::float3& center, float width, float height)
{
    // XZ-plane rectangle whose shader normal (-cross(p1-p0, p3-p0)) points -Y,
    // i.e. a ceiling light facing the room below.
    RectCorners c;
    const float hw = 0.5f * width;
    const float hh = 0.5f * height;
    c.p0 = center + glm::float3(-hw, 0.0f, hh);
    c.p1 = center + glm::float3(hw, 0.0f, hh);
    c.p3 = center + glm::float3(-hw, 0.0f, -hh);
    return c;
}

inline float rectArea(const RectCorners& c)
{
    const glm::float3 e1 = c.p1 - c.p0;
    const glm::float3 e2 = c.p3 - c.p0;
    return length3(glm::cross(e1, e2));
}

inline glm::float3 rectNormal(const RectCorners& c)
{
    // Same sign convention as the shaders: -normalize(cross(e1, e2)).
    const glm::float3 e1 = c.p1 - c.p0;
    const glm::float3 e2 = c.p3 - c.p0;
    return -glm::normalize(glm::cross(e1, e2));
}

// Solid angle of a planar rectangle as seen from `o`, via the spherical excess
// (Girard's theorem). Used as an independent check that SphQuad::S is right.
inline float solidAngleGirard(const RectCorners& c, const glm::float3& o)
{
    const glm::float3 v0 = glm::normalize(c.p0 - o);
    const glm::float3 v1 = glm::normalize(c.p1 - o);
    const glm::float3 v2 = glm::normalize(c.p1 + (c.p3 - c.p0) - o); // p2
    const glm::float3 v3 = glm::normalize(c.p3 - o);

    const auto edgeNormal = [](const glm::float3& a, const glm::float3& b) { return glm::normalize(glm::cross(a, b)); };
    const glm::float3 n0 = edgeNormal(v0, v1);
    const glm::float3 n1 = edgeNormal(v1, v2);
    const glm::float3 n2 = edgeNormal(v2, v3);
    const glm::float3 n3 = edgeNormal(v3, v0);

    const float g0 = std::acos(std::clamp(-glm::dot(n0, n1), -1.0f, 1.0f));
    const float g1 = std::acos(std::clamp(-glm::dot(n1, n2), -1.0f, 1.0f));
    const float g2 = std::acos(std::clamp(-glm::dot(n2, n3), -1.0f, 1.0f));
    const float g3 = std::acos(std::clamp(-glm::dot(n3, n0), -1.0f, 1.0f));
    return g0 + g1 + g2 + g3 - 2.0f * 3.14159265358979323846f;
}

// Cycles-hardened SphQuad init, from common/rect_sampling.h. S <= 0 means the
// rectangle is edge-on or behind the local plane of the light.
inline SphQuad initSphQuad(const RectCorners& c, const glm::float3& o)
{
    return sphQuadInit(c.p0, c.p1 - c.p0, c.p3 - c.p0, o);
}

inline glm::float3 sampleSphQuad(const SphQuad& squad, float u, float v)
{
    return sphQuadSample(squad, u, v);
}

inline float areaToSolidAnglePdf(const glm::float3& pointOnLight,
                                 const glm::float3& hitPoint,
                                 const glm::float3& lightNormal,
                                 float area)
{
    const glm::float3 toLight = pointOnLight - hitPoint;
    const float dist = length3(toLight);
    if (dist < 1e-8f || area <= 0.0f)
    {
        return 0.0f;
    }
    const glm::float3 L = toLight / dist;
    const float cosAtLight = glm::dot(-L, lightNormal);
    if (cosAtLight <= 0.0f)
    {
        return 0.0f;
    }
    return (dist * dist) / (cosAtLight * area);
}

inline LightSample sampleRectUniform(const RectCorners& c, const glm::float2& uv, const glm::float3& hitPoint)
{
    LightSample s;
    const glm::float3 e1 = c.p1 - c.p0;
    const glm::float3 e2 = c.p3 - c.p0;
    s.pointOnLight = c.p0 + e1 * uv.x + e2 * uv.y;
    s.area = rectArea(c);
    s.normal = rectNormal(c);
    const glm::float3 toLight = s.pointOnLight - hitPoint;
    s.distToLight = length3(toLight);
    s.L = (s.distToLight > 1e-8f) ? toLight / s.distToLight : glm::float3(0.0f);
    s.pdf = areaToSolidAnglePdf(s.pointOnLight, hitPoint, s.normal, s.area);
    return s;
}

inline LightSample sampleRectSolidAngle(const RectCorners& c, const glm::float2& uv, const glm::float3& hitPoint)
{
    const SphQuad squad = initSphQuad(c, hitPoint);
    if (squad.S <= 0.0f)
    {
        LightSample s = sampleRectUniform(c, uv, hitPoint);
        s.pdf = 0.0f;
        return s;
    }
    if (squad.useAreaFallback)
    {
        return sampleRectUniform(c, uv, hitPoint);
    }

    LightSample s;
    s.pointOnLight = sampleSphQuad(squad, uv.x, uv.y);
    s.area = rectArea(c);
    s.normal = rectNormal(c);
    const glm::float3 toLight = s.pointOnLight - hitPoint;
    s.distToLight = length3(toLight);
    s.L = (s.distToLight > 1e-8f) ? toLight / s.distToLight : glm::float3(0.0f);
    s.pdf = 1.0f / squad.S;
    return s;
}

// PDF that MIS must use for a BSDF hit on this rect, matching the NEE strategy.
inline float rectLightPdf(const RectCorners& c,
                          const glm::float3& lightHitPoint,
                          const glm::float3& surfaceHitPoint,
                          bool solidAngle)
{
    if (!solidAngle)
    {
        return areaToSolidAnglePdf(lightHitPoint, surfaceHitPoint, rectNormal(c), rectArea(c));
    }
    const SphQuad squad = initSphQuad(c, surfaceHitPoint);
    if (squad.S <= 0.0f)
    {
        return 0.0f;
    }
    if (squad.useAreaFallback)
    {
        return areaToSolidAnglePdf(lightHitPoint, surfaceHitPoint, rectNormal(c), rectArea(c));
    }
    return 1.0f / squad.S;
}

} // namespace oka::rect_light_sampling

