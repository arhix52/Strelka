#pragma once
#include <vector_types.h>
#include <sutil/vec_math.h>
#include <light_types.h>

// GPU side structure
// pad0: spot inner cone (rad) or point soft radius.
// pad1: KHR attenuation range (0 = infinite).
// points[0].y for point/spot: IES profile index, or -1 when isotropic.
struct UniformLight
{
    float4 points[4];
    float4 color;
    float4 normal;
    int type;
    float halfAngle;
    float pad0;
    float pad1;
};

// Packed IES candela tables for the GPU. OptiXRender::createIesBuffer lays the
// buffer out as:
//   IesGpuBufferHeader
//   IesGpuProfileHeader[profileCount]
//   float blob (angles then candela, offsets relative to the blob start)
// Sampled by sampleIesCandela() below; intensity on the light is a multiplier
// on top of the table. Field for field this is the Metal ShaderTypes.h pair, so
// a profile packed by either backend reads the same on the other.
struct IesGpuBufferHeader
{
    unsigned int profileCount;
    unsigned int floatOffset; // byte offset of the float blob from the buffer start
    unsigned int pad0;
    unsigned int pad1;
};

struct IesGpuProfileHeader
{
    unsigned int nVertical;
    unsigned int nHorizontal;
    unsigned int anglesOffset;  // index into the float blob: vertical then horizontal
    unsigned int candelaOffset; // index into the float blob
    float maxCandela;
    float pad0;
    float pad1;
    float pad2;
};

struct LightSampleData
{
    float3 pointOnLight;
    float pdf;

    float3 normal;
    float area;

    float3 L;
    float distToLight;
};

__forceinline__ __device__ float misWeightBalance(const float a, const float b)
{
    return 1.0f / ( 1.0f + (b / a) );
}

__forceinline__ __device__ float misWeightPower(const float a, const float b)
{
    const float a2 = a * a;
    return a2 / (a2 + b * b);
}

// Dispatch: 0 = balance heuristic, 1 = power heuristic
__forceinline__ __device__ float computeMisWeight(const float a, const float b, const uint32_t heuristic)
{
    return (heuristic == 1) ? misWeightPower(a, b) : misWeightBalance(a, b);
}

static __inline__ __device__ float calcLightArea(const UniformLight& l)
{
    float area = 0.0f;

    if (l.type == LIGHT_TYPE_RECT)
    {
        float3 e1 = make_float3(l.points[1]) - make_float3(l.points[0]);
        float3 e2 = make_float3(l.points[3]) - make_float3(l.points[0]);
        area = length(cross(e1, e2));
    }
    else if (l.type == LIGHT_TYPE_DISC)
    {
        area = M_PIf * l.points[0].x * l.points[0].x; // pi * radius^2
    }
    else if (l.type == LIGHT_TYPE_SPHERE)
    {
        area = 4.0f * M_PIf * l.points[0].x * l.points[0].x; // 4 * pi * radius^2
    }
    return area;
}

static __inline__ __device__ float3 calcLightNormal(const UniformLight& l, const float3 hitPoint)
{
    float3 norm = make_float3(0.0f);

    if (l.type == LIGHT_TYPE_RECT)
    {
        float3 e1 = make_float3(l.points[1]) - make_float3(l.points[0]);
        float3 e2 = make_float3(l.points[3]) - make_float3(l.points[0]);

        norm = -normalize(cross(e1, e2));
    }
    else if (l.type == LIGHT_TYPE_DISC)
    {
        norm = make_float3(l.normal);
    }
    else if (l.type == LIGHT_TYPE_SPHERE)
    {
        norm = normalize(hitPoint - make_float3(l.points[1]));
    }
    return norm;
}

static __inline__ __device__ void fillLightData(const UniformLight& l, const float3 hitPoint, LightSampleData& lightSampleData)
{
    lightSampleData.area = calcLightArea(l);
    lightSampleData.normal = calcLightNormal(l, hitPoint);
    const float3 toLight = lightSampleData.pointOnLight - hitPoint;
    const float lenToLight = length(toLight);
    lightSampleData.L = toLight / lenToLight;
    lightSampleData.distToLight = lenToLight;
}

struct SphQuad
{
    float3 o, x, y, z;
    float z0;
    float x0, y0;
    float x1, y1;
    float b0, b1, b0sq;
    float g2, g3;
    float S;
    bool useAreaFallback;
};

// Ureña / Fajardo / King EGSR 2013, Cycles-hardened (asin form). See the Metal
// lights_metal.h comment for why acos is not used here.
static __device__ SphQuad init(const UniformLight& l, const float3 o)
{
    SphQuad squad;

    float3 ex = make_float3(l.points[1]) - make_float3(l.points[0]);
    float3 ey = make_float3(l.points[3]) - make_float3(l.points[0]);
    float3 s = make_float3(l.points[0]);

    float exl = length(ex);
    float eyl = length(ey);

    squad.o = o;
    squad.x = ex / exl;
    squad.y = ey / eyl;
    squad.z = cross(squad.x, squad.y);

    float3 d = s - o;
    squad.z0 = dot(d, squad.z);
    if (squad.z0 > 0.0f)
    {
        squad.z *= -1.0f;
        squad.z0 *= -1.0f;
    }

    squad.x0 = dot(d, squad.x);
    squad.y0 = dot(d, squad.y);
    squad.x1 = squad.x0 + exl;
    squad.y1 = squad.y0 + eyl;

    float4 nz = make_float4(-squad.y0, squad.x1, squad.y1, -squad.x0);
    nz.x /= sqrtf(nz.x * nz.x + squad.z0 * squad.z0);
    nz.y /= sqrtf(nz.y * nz.y + squad.z0 * squad.z0);
    nz.z /= sqrtf(nz.z * nz.z + squad.z0 * squad.z0);
    nz.w /= sqrtf(nz.w * nz.w + squad.z0 * squad.z0);

    float g0 = asinf(fminf(fmaxf(-nz.x * nz.y, -1.0f), 1.0f));
    float g1 = asinf(fminf(fmaxf(-nz.y * nz.z, -1.0f), 1.0f));
    float g2 = asinf(fminf(fmaxf(-nz.z * nz.w, -1.0f), 1.0f));
    float g3 = asinf(fminf(fmaxf(-nz.w * nz.x, -1.0f), 1.0f));
    squad.S = -(g0 + g1 + g2 + g3);
    squad.g2 = g2;
    squad.g3 = g3;
    squad.b0 = nz.x;
    squad.b1 = nz.z;
    squad.b0sq = squad.b0 * squad.b0;

    const float nzMinSq = fminf(fminf(nz.x * nz.x, nz.y * nz.y), fminf(nz.z * nz.z, nz.w * nz.w));
    squad.useAreaFallback = (squad.S < 1e-5f) || (nzMinSq > 0.99999f);
    return squad;
}

static __device__ float3 SphQuadSample(const SphQuad& squad, const float2 uv)
{
    float u = uv.x;
    float v = uv.y;

    float au = u * squad.S + squad.g2 + squad.g3;
    float sinAu = sinf(au);
    float fu = (fabsf(sinAu) > 1e-8f) ? (cosf(au) * squad.b0 + squad.b1) / sinAu : 0.0f;
    float cu = copysignf(1.0f / sqrtf(fu * fu + squad.b0sq), fu);
    cu = clamp(cu, -1.0f, 1.0f);

    float xu = -(cu * squad.z0) / fmaxf(sqrtf(1.0f - cu * cu), 1e-7f);
    xu = clamp(xu, squad.x0, squad.x1);

    float d2 = xu * xu + squad.z0 * squad.z0;
    float h0 = squad.y0 / sqrtf(d2 + squad.y0 * squad.y0);
    float h1 = squad.y1 / sqrtf(d2 + squad.y1 * squad.y1);
    float hv = h0 + v * (h1 - h0);
    float hv2 = hv * hv;
    float yv = (hv2 < 1.0f - 1e-6f) ? hv * sqrtf(d2 / (1.0f - hv2)) : squad.y1;

    return (squad.o + xu * squad.x + yv * squad.y + squad.z0 * squad.z);
}

// MIS needs only the solid angle. Keep the full SphQuad out of that path so the
// sample basis and inverse-CDF constants do not consume registers on light hits.
static __inline__ __device__ float rectSolidAngle(const UniformLight& l,
                                                  const float3 o,
                                                  bool& useAreaFallback)
{
    const float3 ex = make_float3(l.points[1]) - make_float3(l.points[0]);
    const float3 ey = make_float3(l.points[3]) - make_float3(l.points[0]);
    const float exl = length(ex);
    const float eyl = length(ey);
    const float3 x = ex / exl;
    const float3 y = ey / eyl;
    const float3 z = cross(x, y);
    const float3 d = make_float3(l.points[0]) - o;
    const float z0 = fabsf(dot(d, z));
    const float x0 = dot(d, x);
    const float y0 = dot(d, y);

    float4 nz = make_float4(-y0, x0 + exl, y0 + eyl, -x0);
    nz.x /= sqrtf(nz.x * nz.x + z0 * z0);
    nz.y /= sqrtf(nz.y * nz.y + z0 * z0);
    nz.z /= sqrtf(nz.z * nz.z + z0 * z0);
    nz.w /= sqrtf(nz.w * nz.w + z0 * z0);

    const float g0 = asinf(fminf(fmaxf(-nz.x * nz.y, -1.0f), 1.0f));
    const float g1 = asinf(fminf(fmaxf(-nz.y * nz.z, -1.0f), 1.0f));
    const float g2 = asinf(fminf(fmaxf(-nz.z * nz.w, -1.0f), 1.0f));
    const float g3 = asinf(fminf(fmaxf(-nz.w * nz.x, -1.0f), 1.0f));
    const float S = -(g0 + g1 + g2 + g3);
    const float nzMinSq =
        fminf(fminf(nz.x * nz.x, nz.y * nz.y), fminf(nz.z * nz.z, nz.w * nz.w));
    useAreaFallback = (S < 1e-5f) || (nzMinSq > 0.99999f);
    return S;
}

// Area-to-solid-angle pdf of a point sampled uniformly on a flat light.
static __inline__ __device__ float getAreaLightPdf(const UniformLight& l, const float3 lightHitPoint, const float3 surfaceHitPoint)
{
    LightSampleData lightSampleData {};
    lightSampleData.pointOnLight = lightHitPoint;
    fillLightData(l, surfaceHitPoint, lightSampleData);
    lightSampleData.pdf = lightSampleData.distToLight * lightSampleData.distToLight /
                            (dot(-lightSampleData.L, lightSampleData.normal) * lightSampleData.area);
    return lightSampleData.pdf;
}

static __inline__ __device__ bool emitsLight(const float3 radiance)
{
    return radiance.x > 0.0f || radiance.y > 0.0f || radiance.z > 0.0f;
}

static __inline__ __device__ float getDirectLightPdf(float angle)
{
    return 1.0f / (2.0f * M_PIf * (1.0f - cos(angle)));
}

static __inline__ __device__ float getSphereLightPdf()
{
    return 1.0f / (4.0f * M_PIf);
}

static __inline__ __device__ float getRectLightPdf(const UniformLight& l,
                                                   const float3 lightHitPoint,
                                                   const float3 surfaceHitPoint,
                                                   unsigned int rectLightSamplingMethod)
{
    if (rectLightSamplingMethod == 0)
    {
        return getAreaLightPdf(l, lightHitPoint, surfaceHitPoint);
    }
    bool useAreaFallback = false;
    const float S = rectSolidAngle(l, surfaceHitPoint, useAreaFallback);
    if (S <= 0.0f)
    {
        return 0.0f;
    }
    if (useAreaFallback)
    {
        return getAreaLightPdf(l, lightHitPoint, surfaceHitPoint);
    }
    return 1.0f / S;
}

static __inline__ __device__ float getLightPdf(const UniformLight& l,
                                               const float3 lightHitPoint,
                                               const float3 surfaceHitPoint,
                                               unsigned int rectLightSamplingMethod = 0)
{
    switch (l.type)
    {
    case LIGHT_TYPE_RECT:
        return getRectLightPdf(l, lightHitPoint, surfaceHitPoint, rectLightSamplingMethod);
    case LIGHT_TYPE_DISC:
        return getAreaLightPdf(l, lightHitPoint, surfaceHitPoint);
    case LIGHT_TYPE_SPHERE:
        return getSphereLightPdf();
    case LIGHT_TYPE_DISTANT:
        return getDirectLightPdf(l.halfAngle);
    case LIGHT_TYPE_DOME:
        return getSphereLightPdf();
    case LIGHT_TYPE_POINT:
    case LIGHT_TYPE_SPOT:
        if (l.points[0].x > 1e-4f)
            return getSphereLightPdf();
        return 1.0f;
    default:
        break;
    }
    return 0.0f;
}

static __inline__ __device__ LightSampleData SampleRectLight(const UniformLight& l, const float2 u, const float3 hitPoint)
{
    LightSampleData lightSampleData;
    float3 e1 = make_float3(l.points[1]) - make_float3(l.points[0]);
    float3 e2 = make_float3(l.points[3]) - make_float3(l.points[0]);
    // lightSampleData.pointOnLight = make_float3(l.points[0]) + e1 * u.x + e2 * u.y;
    // https://www.arnoldrenderer.com/research/egsr2013_spherical_rectangle.pdf
    SphQuad quad = init(l, hitPoint);
    if (quad.S <= 0.0f)
    {
        lightSampleData.pdf = 0.0f;
        lightSampleData.pointOnLight = make_float3(l.points[0]) + e1 * u.x + e2 * u.y;
        fillLightData(l, hitPoint, lightSampleData);
        return lightSampleData;
    }
    if (quad.useAreaFallback)
    {
        lightSampleData.pointOnLight = make_float3(l.points[0]) + e1 * u.x + e2 * u.y;
        fillLightData(l, hitPoint, lightSampleData);
        lightSampleData.pdf = lightSampleData.distToLight * lightSampleData.distToLight /
                              (-dot(lightSampleData.L, lightSampleData.normal) * lightSampleData.area);
        return lightSampleData;
    }

    lightSampleData.pointOnLight = SphQuadSample(quad, u);
    fillLightData(l, hitPoint, lightSampleData);
    lightSampleData.pdf = 1.0f / quad.S;

    return lightSampleData;
}

static __inline__ __device__ LightSampleData SampleRectLightUniform(const UniformLight& l, const float2 u, const float3 hitPoint)
{
    LightSampleData lightSampleData;
    // // uniform sampling
    float3 e1 = make_float3(l.points[1]) - make_float3(l.points[0]);
    float3 e2 = make_float3(l.points[3]) - make_float3(l.points[0]);
    lightSampleData.pointOnLight = make_float3(l.points[0]) + e1 * u.x + e2 * u.y;
    fillLightData(l, hitPoint, lightSampleData);
    // here is conversion from are to solid angle: dist2 / cos
    lightSampleData.pdf = lightSampleData.distToLight * lightSampleData.distToLight /
                          (-dot(lightSampleData.L, lightSampleData.normal) * lightSampleData.area);
    return lightSampleData;
}

static __device__ void createCoordinateSystem(const float3& N, float3& Nt, float3& Nb) {
    if (fabs(N.x) > fabs(N.y)) {
        float invLen = 1.0f / sqrt(N.x * N.x + N.z * N.z);
        Nt = make_float3(-N.z * invLen, 0.0f, N.x * invLen);
    } else {
        float invLen = 1.0f / sqrt(N.y * N.y + N.z * N.z);
        Nt = make_float3(0.0f, N.z * invLen, -N.y * invLen);
    }
    Nb = cross(N, Nt);
}

static __device__ float3 SampleCone(float2 uv, float angle, float3 direction, float& pdf) {

    float phi = 2.0 * M_PIf * uv.x;
    const float halfSin = sin(0.5f * angle);
    float cosTheta = 1.0 - uv.y * (2.0f * halfSin * halfSin);

    // Convert spherical coordinates to 3D direction
    float sinTheta = sqrt(1.0 - cosTheta * cosTheta);

    float3 u, v;
    createCoordinateSystem(direction, u, v);
    float3 sampledDir = normalize(cos(phi) * sinTheta * u + sin(phi) * sinTheta * v + cosTheta * direction);

    // Calculate the PDF for the sampled direction
    // 4pi sin^2(x/2), not 2pi (1 - cos x): see coneSolidAngle() in light_desc.h.
    // The host bakes a distant light's radiance as irradiance / solid angle and
    // this divides it back out, so the two have to be the same number; at
    // sun-sized angles 1 - cos is mostly rounding.
    pdf = 1.0f / (4.0f * M_PIf * halfSin * halfSin);
    return sampledDir;
}

static __inline__ __device__ LightSampleData SampleDistantLight(const UniformLight& l, const float2 u, const float3 hitPoint)
{
    LightSampleData lightSampleData;
    float pdf = 0.0f;
    float3 coneSample = SampleCone(u, l.halfAngle, -make_float3(l.normal), pdf);

    lightSampleData.area = 0.0f;
    lightSampleData.distToLight = 1e9;
    lightSampleData.L = coneSample;
    lightSampleData.normal = make_float3(l.normal);
    lightSampleData.pdf = pdf;
    lightSampleData.pointOnLight = coneSample;

    return lightSampleData;
}

// Uniform over the disc's area. points[0].x carries the radius and points[2..3]
// the in-plane axes; the axes are normalized here because whether the light's
// transform already scaled them depends on how the light was authored, while the
// area is computed from the radius in points[0] either way.
static __inline__ __device__ LightSampleData SampleDiscLight(const UniformLight& l, const float2 u, const float3 hitPoint)
{
    LightSampleData lightSampleData;

    const float radius = l.points[0].x;
    const float3 center = make_float3(l.points[1]);
    const float3 axisX = normalize(make_float3(l.points[2]));
    const float3 axisY = normalize(make_float3(l.points[3]));

    // sqrt keeps the samples uniform per unit area rather than crowding the centre.
    const float r = radius * sqrtf(u.x);
    const float phi = 2.0f * M_PIf * u.y;
    lightSampleData.pointOnLight = center + r * (cosf(phi) * axisX + sinf(phi) * axisY);

    fillLightData(l, hitPoint, lightSampleData);
    const float cosAtLight = -dot(lightSampleData.L, lightSampleData.normal);
    lightSampleData.pdf = (cosAtLight > 0.0f && lightSampleData.area > 0.0f)
                              ? lightSampleData.distToLight * lightSampleData.distToLight /
                                    (cosAtLight * lightSampleData.area)
                              : 0.0f;
    return lightSampleData;
}

static __inline__ __device__ LightSampleData SampleSphereLight(const UniformLight& l, const float2 u, const float3 hitPoint)
{
    LightSampleData lightSampleData;

    // Generate a random direction on the sphere using solid angle sampling
    float cosTheta = 1.0f - 2.0f * u.x;  // cosTheta is uniformly distributed between [-1, 1]
    float sinTheta = sqrt(1.0f - cosTheta * cosTheta);
    float phi = 2.0f * M_PIf * u.y;  // phi is uniformly distributed between [0, 2*pi]

    const float radius = l.points[0].x;

    // Convert spherical coordinates to Cartesian coordinates
    float3 sphereDirection = make_float3(sinTheta * cos(phi), sinTheta * sin(phi), cosTheta);
    // Scale the direction by the radius of the sphere and move it to the light position
    float3 lightPoint = make_float3(l.points[1]) + radius * sphereDirection;
    // Calculate the direction from the hit point to the sampled point on the light
    lightSampleData.L = normalize(lightPoint - hitPoint);

    // Calculate the distance to the light
    lightSampleData.distToLight = length(lightPoint - hitPoint);

    lightSampleData.area = 0.0f;
    lightSampleData.normal = sphereDirection;
    lightSampleData.pdf = 1.0f / (4.0f * M_PIf);
    lightSampleData.pointOnLight = lightPoint;

    return lightSampleData;
}

/// An infinitely distant, uniform-radiance dome.
///
/// Uniform over the whole sphere rather than the upper hemisphere: a dome is the
/// analytic form of an environment, and an environment lights a surface from
/// below as well as above once anything reflects. `color` is radiance, so there
/// is no distance falloff and no area -- the pdf is the constant 1/4pi, which is
/// what getLightPdf() returns for this type so that MIS against a BSDF ray that
/// misses the scene agrees with what was sampled here.
///
/// Without this case the switch in sampleLight fell through, leaving a
/// zero-initialised LightSampleData: direction (0,0,0), pdf 0. The facing test
/// then rejected it, so a dome light contributed exactly nothing and did so
/// silently -- no NaN, no red pixel, just an unlit scene.
static __inline__ __device__ LightSampleData SampleDomeLight(const UniformLight& l, const float2 u, const float3 hitPoint)
{
    LightSampleData lightSampleData;

    const float cosTheta = 1.0f - 2.0f * u.x; // uniform on [-1, 1]
    const float sinTheta = sqrtf(fmaxf(1.0f - cosTheta * cosTheta, 0.0f));
    const float phi = 2.0f * M_PIf * u.y;

    lightSampleData.L = make_float3(sinTheta * cosf(phi), cosTheta, sinTheta * sinf(phi));
    lightSampleData.distToLight = 1e9f;
    lightSampleData.area = 0.0f;
    // Faces the shading point by construction, so the caller's -dot(L, normal)
    // test passes for every sampled direction.
    lightSampleData.normal = -lightSampleData.L;
    lightSampleData.pdf = 1.0f / (4.0f * M_PIf);
    lightSampleData.pointOnLight = hitPoint + lightSampleData.L * lightSampleData.distToLight;

    return lightSampleData;
}

static __inline__ __device__ LightSampleData SamplePointLight(const UniformLight& l, const float2 u, const float3 hitPoint)
{
    const float radius = l.points[0].x;
    if (radius > 1e-4f)
    {
        return SampleSphereLight(l, u, hitPoint);
    }

    LightSampleData lightSampleData;
    const float3 center = make_float3(l.points[1]);
    const float3 toLight = center - hitPoint;
    const float dist = length(toLight);
    lightSampleData.pointOnLight = center;
    lightSampleData.L = toLight / fmaxf(dist, 1e-8f);
    lightSampleData.distToLight = dist;
    lightSampleData.normal = -lightSampleData.L;
    lightSampleData.area = 0.0f;
    lightSampleData.pdf = 1.0f;
    return lightSampleData;
}

static __inline__ __device__ float spotAttenuation(const UniformLight& l, const float3 dirFromLight)
{
    const float3 axis = normalize(make_float3(l.normal));
    const float cosOuter = cosf(l.halfAngle);
    const float cosInner = cosf(l.pad0);
    const float cosTheta = dot(axis, dirFromLight);
    if (cosTheta < cosOuter)
        return 0.0f;
    if (cosInner <= cosOuter)
        return 1.0f;
    // clamp(), not saturate(): saturate lives in sutil/vec_math_adv.h, which is
    // written for nvcc and does not compile in a plain host translation unit --
    // and this header is reached from host code through OptixRenderParams.h.
    // clamp() comes from vec_math.h, which is already included above.
    return clamp((cosTheta - cosOuter) / (cosInner - cosOuter), 0.0f, 1.0f);
}

static __inline__ __device__ float rangeWindow(const UniformLight& l, float dist)
{
    if (l.pad1 <= 0.0f)
        return 1.0f;
    const float x = clamp(dist / l.pad1, 0.0f, 1.0f);
    const float y = 1.0f - x * x * x * x;
    return y * y;
}

// Index of the interval containing x in the ascending table a, clamped so both
// it and it+1 are addressable. Binary search rather than a linear scan because a
// real luminaire's vertical table runs to 181 entries and this is evaluated once
// per shadow connection.
static __inline__ __device__ int iesLowerIndex(const float* a, int n, float x)
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
    if (lo <= 0)
    {
        return 0;
    }
    if (lo >= n)
    {
        return max(0, n - 2);
    }
    return lo - 1;
}

// Bilinear sample of an IES candela table. `dirFromLight` is world-space; the
// light's local frame is rebuilt from points[2..3] (X/Y axes) and normal (-Z),
// the same packing Scene::updateLight writes for the CPU sampler in
// iesloader.cpp. Returns 1.0 when the light carries no profile, so the caller
// can multiply unconditionally.
static __inline__ __device__ float sampleIesCandela(const IesGpuBufferHeader* iesBuffer,
                                                    const UniformLight& l,
                                                    const float3 dirFromLight)
{
    const int profileIdx = (int)l.points[0].y;
    if (!iesBuffer || profileIdx < 0 || (unsigned int)profileIdx >= iesBuffer->profileCount)
    {
        return 1.0f;
    }

    const IesGpuProfileHeader* headers =
        (const IesGpuProfileHeader*)((const char*)iesBuffer + sizeof(IesGpuBufferHeader));
    const IesGpuProfileHeader& h = headers[profileIdx];
    if (h.nVertical < 2u || h.nHorizontal < 1u)
    {
        return 0.0f;
    }

    const float* floats = (const float*)((const char*)iesBuffer + iesBuffer->floatOffset);

    // World -> light local. Columns of the light's basis; -Z is the photometric
    // axis, matching iesloader.cpp::sampleIesCandela.
    const float3 ax = normalize(make_float3(l.points[2]));
    const float3 ay = normalize(make_float3(l.points[3]));
    const float3 az = normalize(make_float3(l.normal)); // emission -Z
    const float3 d = normalize(dirFromLight);
    const float3 local = make_float3(dot(d, ax), dot(d, ay), -dot(d, az));

    const float vertDeg = acosf(clamp(-local.z, -1.0f, 1.0f)) * (180.0f / M_PIf);
    float horizDeg = atan2f(local.x, -local.y) * (180.0f / M_PIf);
    if (horizDeg < 0.0f)
    {
        horizDeg += 360.0f;
    }

    const float* vAng = floats + h.anglesOffset;
    const float* hAng = floats + h.anglesOffset + h.nVertical;
    const float* candela = floats + h.candelaOffset;
    const int nV = (int)h.nVertical;
    const int nH = (int)h.nHorizontal;

    const int iv = max(0, min(nV - 2, iesLowerIndex(vAng, nV, vertDeg)));
    int ih = 0;
    float th = 0.0f;
    if (nH > 1)
    {
        // A table that stops at 90 or 180 degrees is stored for one symmetric
        // quadrant or half; fold the azimuth back into the range it covers.
        float hDeg = horizDeg;
        const float hMax = hAng[nH - 1];
        if (hMax <= 90.0f + 1e-3f)
        {
            hDeg = fmodf(hDeg, 90.0f);
        }
        else if (hMax <= 180.0f + 1e-3f)
        {
            if (hDeg > 180.0f)
            {
                hDeg = 360.0f - hDeg;
            }
        }
        else
        {
            hDeg = fmodf(hDeg, 360.0f);
        }
        ih = max(0, min(nH - 2, iesLowerIndex(hAng, nH, hDeg)));
        const float h0 = hAng[ih];
        const float h1 = hAng[ih + 1];
        th = (h1 > h0) ? (hDeg - h0) / (h1 - h0) : 0.0f;
    }

    const float v0 = vAng[iv];
    const float v1 = vAng[iv + 1];
    const float tv = (v1 > v0) ? (vertDeg - v0) / (v1 - v0) : 0.0f;

    const int ih1 = (nH == 1) ? 0 : ih + 1;
    const float c00 = candela[iv + ih * nV];
    const float c10 = candela[(iv + 1) + ih * nV];
    const float c01 = candela[iv + ih1 * nV];
    const float c11 = candela[(iv + 1) + ih1 * nV];
    const float c0 = c00 * (1.0f - tv) + c10 * tv;
    const float c1 = c01 * (1.0f - tv) + c11 * tv;
    return c0 * (1.0f - th) + c1 * th;
}
