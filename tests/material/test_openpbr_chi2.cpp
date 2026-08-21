// ============================================================================
// test_openpbr_chi2.cpp
//
// Does the sampler draw from the density it reports?
//
// test_openpbr_consistency.cpp already checks sample and eval against each other
// at the directions the sampler produced. That is a sharp test and it has one
// blind spot, which is the reason this file exists: it only ever looks where the
// sampler chose to look. A sampler and a pdf that are wrong by the same factor,
// or that agree pointwise while the sampler visits the wrong part of the
// hemisphere, pass it and fail here.
//
// The test is the standard one: bin the sphere into equal-solid-angle cells,
// draw many directions, and compare the histogram against the pdf integrated
// over each cell. Bins are uniform in cos(theta) and in phi, so every cell
// subtends the same solid angle and no Jacobian enters the comparison.
//
// Materials are deliberately rough. A near-delta lobe puts almost all of its
// mass inside one cell, and the statistic then measures the binning rather than
// the sampler; delta events are skipped outright for the same reason.
//
// Checked by re-introducing the defect it guards, which for a statistical test
// is the only way to know it is doing anything: multiplying the reported pdf by
// (1 + 0.25 * wi.x) -- a 25% skew across the azimuth, invisible to every other
// test in the suite -- takes the reduced statistic from passing to 14.6 and 14.7
// on two of the four materials, against a bound of 12. That is the sensitivity
// this is worth: it will catch a sampler pointed the wrong way, not a percent.
// ============================================================================

#include <doctest/doctest.h>

#include <strelka/material/bsdf_types.h>
#include <strelka/material/material_math.h>
#include <strelka/material/openpbr/openpbr_bridge.h>
#include <strelka/material/openpbr/openpbr_params.h>

#include "../support/sampling.h"

#include <cmath>
#include <string>
#include <vector>

namespace
{

constexpr int kCosBins = 8; // over [-1, 1]
constexpr int kPhiBins = 16;
constexpr int kBins = kCosBins * kPhiBins;
constexpr int kSamples = 60000;
constexpr int kSubSamples = 4; // per axis, when integrating the pdf over a cell

constexpr float kPi = 3.14159265358979323846f;

int binOf(float3 w)
{
    const float cosTheta = std::fmin(std::fmax(w.z, -1.0f), 1.0f);
    float phi = std::atan2(w.y, w.x);
    if (phi < 0.0f)
    {
        phi += 2.0f * kPi;
    }
    int ci = static_cast<int>((cosTheta + 1.0f) * 0.5f * kCosBins);
    int pi = static_cast<int>(phi / (2.0f * kPi) * kPhiBins);
    ci = std::min(std::max(ci, 0), kCosBins - 1);
    pi = std::min(std::max(pi, 0), kPhiBins - 1);
    return ci * kPhiBins + pi;
}

/// Van der Corput, so the draw is a deterministic sweep rather than a stream:
/// a statistical test that fails one run in twenty is a test people re-run.
float radicalInverse(unsigned int bits, unsigned int base)
{
    float result = 0.0f;
    float f = 1.0f / static_cast<float>(base);
    while (bits > 0u)
    {
        result += static_cast<float>(bits % base) * f;
        bits /= base;
        f /= static_cast<float>(base);
    }
    return result;
}

SurfaceInteraction surfaceLookingAt(float3 wo)
{
    SurfaceInteraction si = {};
    si.tangent = make_float3(1.0f, 0.0f, 0.0f);
    si.bitangent = make_float3(0.0f, 1.0f, 0.0f);
    si.shading_normal = make_float3(0.0f, 0.0f, 1.0f);
    si.geometry_normal = si.shading_normal;
    si.wo = normalize(wo);
    si.exterior_ior = 1.0f;
    si.front_face = true;
    si.material_type = MATERIAL_TYPE_OPENPBR;
    return si;
}

struct Named
{
    std::string name;
    OpenPBRParams params;
};

std::vector<Named> roughLadder()
{
    std::vector<Named> out;
    {
        OpenPBRParams p = openpbr_make_default_params();
        p.specular_roughness = 0.7f;
        out.push_back({ "rough dielectric", p });
    }
    {
        OpenPBRParams p = openpbr_make_default_params();
        p.base_metalness = 1.0f;
        p.specular_roughness = 0.6f;
        out.push_back({ "rough metal", p });
    }
    {
        OpenPBRParams p = openpbr_make_default_params();
        p.specular_roughness = 0.6f;
        p.transmission_weight = 1.0f;
        out.push_back({ "rough transmission", p });
    }
    {
        OpenPBRParams p = openpbr_make_default_params();
        p.specular_roughness = 0.5f;
        p.fuzz_weight = 1.0f;
        p.fuzz_roughness = 0.5f;
        out.push_back({ "fuzz over diffuse", p });
    }
    return out;
}

} // namespace

TEST_CASE("openpbr: the histogram of sampled directions matches the reported pdf")
{
    for (const Named& material : roughLadder())
    {
        CAPTURE(material.name);

        const SurfaceInteraction si = surfaceLookingAt(make_float3(0.4f, 0.1f, 0.9f));
        const OpenPBR_PreparedBsdf prepared = openpbr_prepare_at(material.params, si, make_float3(1.0f));

        // Observed.
        std::vector<double> observed(kBins, 0.0);
        int kept = 0;
        int deltaHits = 0;
        for (int i = 0; i < kSamples; ++i)
        {
            const unsigned int n = static_cast<unsigned int>(i);
            const float4 xi = make_float4(radicalInverse(n, 2), radicalInverse(n, 3), radicalInverse(n, 5), 0.5f);
            const BsdfSampleResult s = openpbr_bsdf_sample(prepared, xi);
            if (s.event_type == BSDF_EVENT_ABSORB || !(s.pdf > 0.0f))
            {
                continue;
            }
            if ((s.event_type & BSDF_EVENT_SPECULAR) != 0u)
            {
                ++deltaHits;
                continue;
            }
            observed[static_cast<size_t>(binOf(s.wi))] += 1.0;
            ++kept;
        }
        REQUIRE(kept > kSamples / 4);
        // A rough material must not be producing delta events.
        CHECK(deltaHits * 20 < kSamples);

        // Expected: the pdf integrated over each cell, by a small regular grid
        // inside it. Every cell has the same solid angle, 4*pi / kBins.
        const double cellSolidAngle = 4.0 * kPi / kBins;
        std::vector<double> expected(kBins, 0.0);
        double total = 0.0;
        for (int ci = 0; ci < kCosBins; ++ci)
        {
            for (int pi = 0; pi < kPhiBins; ++pi)
            {
                double sum = 0.0;
                for (int a = 0; a < kSubSamples; ++a)
                {
                    for (int b = 0; b < kSubSamples; ++b)
                    {
                        const float u = (static_cast<float>(ci) + oka::test::stratum(a, kSubSamples)) / kCosBins;
                        const float v = (static_cast<float>(pi) + oka::test::stratum(b, kSubSamples)) / kPhiBins;
                        const float cosTheta = 2.0f * u - 1.0f;
                        const float sinTheta = std::sqrt(std::fmax(0.0f, 1.0f - cosTheta * cosTheta));
                        const float phi = 2.0f * kPi * v;
                        const float3 w = make_float3(sinTheta * std::cos(phi), sinTheta * std::sin(phi), cosTheta);
                        sum += openpbr_bsdf_pdf(prepared, w);
                    }
                }
                const double mean = sum / (kSubSamples * kSubSamples);
                const double mass = mean * cellSolidAngle;
                const size_t cell = static_cast<size_t>(ci) * kPhiBins + static_cast<size_t>(pi);
                expected[cell] = mass;
                total += mass;
            }
        }
        REQUIRE(total > 0.5); // the density has to integrate to something

        // Compared as shapes: the sampler may legitimately reject a fraction of
        // its draws, so what is under test is where the mass went, not how much
        // of it there was.
        double chi2 = 0.0;
        int cells = 0;
        for (int i = 0; i < kBins; ++i)
        {
            const double e = expected[static_cast<size_t>(i)] / total * kept;
            if (e < 5.0)
            {
                continue; // the statistic is not valid on a nearly empty cell
            }
            const double d = observed[static_cast<size_t>(i)] - e;
            chi2 += d * d / e;
            ++cells;
        }
        REQUIRE(cells > 10);

        // Reduced chi-square. One is the ideal; the bound is loose because the
        // expected counts come from a 4x4 quadrature of a lobe that is not flat
        // inside a cell, which biases the comparison on its own.
        const double reduced = chi2 / cells;
        CAPTURE(cells);
        CAPTURE(reduced);
        CHECK(reduced < 12.0);
    }
}
