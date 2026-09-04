#include <doctest/doctest.h>

// The frustum maths both backends compile into their shaders. Scalars in, place
// in the image out -- see the note at the top of projector.h for why the texture
// fetch is not here and cannot be.
#include <projector.h>
#include <host/projector_transfer.h>

#include <algorithm>
#include <array>
#include <filesystem>
#include <fstream>

// The host mirror of projectorSolidAngle(), which bakes a projector's Watts.
// The last case in this file is what keeps the copy honest.
#include <strelka/scene/glm_wrapper.hpp>
#include <strelka/scene/light_desc.h>

#include <cmath>
#include <limits>
#include <numbers>

#if defined(__APPLE__)
#    include <MetalTextures.h>
#    include <Foundation/Foundation.hpp>
#    include <unistd.h>
#endif

static constexpr float kPi = std::numbers::pi_v<float>;

// A 90 degree horizontal field on a square frame: the pyramid whose apex angle
// makes six of it fill the sphere. Used by several cases below.
static constexpr float kHalfFov90 = 0.25f * kPi;

TEST_CASE("the beam axis lands in the middle of the frame")
{
    const float tx = projectorTanHalfX(kHalfFov90);
    const float ty = projectorTanHalfY(tx, 1.0f);
    const ProjectorSample s = projectorProject(0.0f, 0.0f, 1.0f, tx, ty, 0.0f);

    CHECK(s.inside);
    CHECK(s.u == doctest::Approx(0.5f));
    CHECK(s.v == doctest::Approx(0.5f));
    CHECK(s.falloff == doctest::Approx(1.0f));
}

TEST_CASE("the frame is upright: +X is to the right and +Y is the top row")
{
    const float tx = projectorTanHalfX(kHalfFov90);
    const float ty = projectorTanHalfY(tx, 1.0f);

    // Half way to the right edge at unit depth.
    const ProjectorSample right = projectorProject(0.5f * tx, 0.0f, 1.0f, tx, ty, 0.0f);
    CHECK(right.inside);
    CHECK(right.u == doctest::Approx(0.75f));
    CHECK(right.v == doctest::Approx(0.5f));

    // The light's up axis has to reach the *top* of the image, which is v = 0:
    // a decoder hands back the top row first. Get this backwards and the image
    // is not obviously wrong -- it is a projector mounted upside down, which is
    // a thing people do on purpose.
    const ProjectorSample up = projectorProject(0.0f, 0.5f * ty, 1.0f, tx, ty, 0.0f);
    CHECK(up.inside);
    CHECK(up.v == doctest::Approx(0.25f));
}

TEST_CASE("a direction outside the rectangle throws nothing")
{
    const float tx = projectorTanHalfX(kHalfFov90);
    const float ty = projectorTanHalfY(tx, 1.0f);

    // Just past the corner in x, still inside in y.
    const ProjectorSample out = projectorProject(1.01f * tx, 0.0f, 1.0f, tx, ty, 0.0f);
    CHECK_FALSE(out.inside);
    CHECK(out.falloff == doctest::Approx(0.0f));

    // Exactly on the border is still the image: the sampler clamps there, and a
    // strict test would leave a one-texel gap at every edge.
    const ProjectorSample edge = projectorProject(tx, 0.0f, 1.0f, tx, ty, 0.0f);
    CHECK(edge.inside);
    CHECK(edge.u == doctest::Approx(1.0f));
}

TEST_CASE("nothing is thrown backwards out of the lens")
{
    const float tx = projectorTanHalfX(kHalfFov90);
    const float ty = projectorTanHalfY(tx, 1.0f);

    // Behind the projector. Not a smooth term that happens to reach zero -- the
    // perspective divide would fold this direction onto a perfectly plausible
    // place in the image, and the light would throw a mirrored copy of the frame
    // out of its own back.
    CHECK_FALSE(projectorProject(0.0f, 0.0f, -1.0f, tx, ty, 0.0f).inside);
    CHECK_FALSE(projectorProject(0.1f, 0.1f, -1.0f, tx, ty, 0.0f).inside);
    // Exactly sideways, where the divide is a division by zero.
    CHECK_FALSE(projectorProject(1.0f, 0.0f, 0.0f, tx, ty, 0.0f).inside);
}

TEST_CASE("the aspect ratio shapes the frame, and 16:9 is wider than it is tall")
{
    const float tx = projectorTanHalfX(kHalfFov90);
    const float ty = projectorTanHalfY(tx, 16.0f / 9.0f);
    CHECK(ty == doctest::Approx(tx * 9.0f / 16.0f));

    // A direction that clears the vertical edge of a square frame is outside a
    // 16:9 one at the same field of view.
    const float y = 0.9f * tx;
    CHECK(projectorProject(0.0f, y, 1.0f, tx, tx, 0.0f).inside);
    CHECK_FALSE(projectorProject(0.0f, y, 1.0f, tx, ty, 0.0f).inside);
}

TEST_CASE("the frame grows linearly with throw distance")
{
    // What makes a projector a projector: the image on a wall twice as far away
    // is twice as wide, and the same texel is at the same place in it.
    const float tx = projectorTanHalfX(0.3f);
    const float ty = projectorTanHalfY(tx, 16.0f / 9.0f);

    const ProjectorSample near = projectorProject(0.4f * tx, 0.2f * ty, 1.0f, tx, ty, 0.0f);
    const ProjectorSample far = projectorProject(0.8f * tx, 0.4f * ty, 2.0f, tx, ty, 0.0f);
    CHECK(near.inside);
    CHECK(far.inside);
    CHECK(far.u == doctest::Approx(near.u));
    CHECK(far.v == doctest::Approx(near.v));
}

TEST_CASE("edge softness fades the border and leaves the middle alone")
{
    const float tx = projectorTanHalfX(kHalfFov90);
    const float ty = projectorTanHalfY(tx, 1.0f);
    const float softness = 0.25f;

    // Well inside the feathered band: untouched.
    CHECK(projectorProject(0.5f * tx, 0.0f, 1.0f, tx, ty, softness).falloff == doctest::Approx(1.0f));
    // At the border: gone.
    CHECK(projectorProject(tx, 0.0f, 1.0f, tx, ty, softness).falloff == doctest::Approx(0.0f));
    // Half way through the band: the smoothstep's midpoint.
    CHECK(projectorProject(0.875f * tx, 0.0f, 1.0f, tx, ty, softness).falloff == doctest::Approx(0.5f));

    // Zero softness is a crisp edge, which is what a focused projector has.
    CHECK(projectorProject(0.999f * tx, 0.0f, 1.0f, tx, ty, 0.0f).falloff == doctest::Approx(1.0f));
}

TEST_CASE("six square pyramids of 90 degrees fill the sphere")
{
    // The one solid angle with an answer that can be checked without trusting
    // the formula: the six faces of a cube seen from its centre partition 4pi.
    const float tx = projectorTanHalfX(kHalfFov90);
    const float omega = projectorSolidAngle(tx, projectorTanHalfY(tx, 1.0f));
    CHECK(6.0f * omega == doctest::Approx(4.0f * kPi).epsilon(1e-5));
}

TEST_CASE("a narrow pyramid approaches the product of its angular extents")
{
    // Small angles: Omega -> 4 tan(a) tan(b), the area of the frame at unit
    // depth. A cone's solid angle is not this number, which is the whole reason
    // projectorSolidAngle() exists.
    const float tx = projectorTanHalfX(0.02f);
    const float ty = projectorTanHalfY(tx, 16.0f / 9.0f);
    CHECK(projectorSolidAngle(tx, ty) == doctest::Approx(4.0f * tx * ty).epsilon(1e-4));

    const float cone = 4.0f * kPi * std::sin(0.01f) * std::sin(0.01f);
    CHECK(cone > 1.35f * projectorSolidAngle(tx, ty));
}

TEST_CASE("the host's solid angle is the shader's, to the last bit")
{
    // oka::projectorSolidAngleFromFov() divides a projector's Watts in
    // scene/light_desc.h and projectorSolidAngle() is what the shader spreads
    // the image over. They are two copies of one number -- the scene header
    // cannot include the shader one, see the comment on the host copy -- so a
    // drift between them scales every projector in the scene and nothing else
    // would catch it.
    const float aspects[] = { 1.0f, 4.0f / 3.0f, 16.0f / 9.0f, 2.39f, 0.75f };
    const float halfFovs[] = { 0.01f, 0.1f, 0.3f, kHalfFov90, 1.5f };
    for (const float aspect : aspects)
    {
        for (const float halfFov : halfFovs)
        {
            const float tx = projectorTanHalfX(halfFov);
            const float shader = projectorSolidAngle(tx, projectorTanHalfY(tx, aspect));
            CHECK(oka::projectorSolidAngleFromFov(halfFov, aspect) == doctest::Approx(shader));
        }
    }
}

TEST_CASE("a degenerate field of view does not produce infinity or zero area")
{
    // 90 degrees is where the tangent blows up and the pyramid stops being one.
    // Clamped rather than guarded at every call site, so the bake divides by
    // something finite whatever a sidecar or a slider hands over.
    const float wide = projectorTanHalfX(0.5f * kPi);
    CHECK(std::isfinite(wide));
    CHECK(std::isfinite(projectorSolidAngle(wide, projectorTanHalfY(wide, 1.0f))));
    CHECK(projectorSolidAngle(wide, projectorTanHalfY(wide, 1.0f)) > 0.0f);

    const float narrow = projectorTanHalfX(0.0f);
    CHECK(narrow > 0.0f);
    CHECK(projectorSolidAngle(narrow, projectorTanHalfY(narrow, 1.0f)) > 0.0f);
}

TEST_CASE("projector LDR texels use the sRGB transfer on every backend")
{
    const std::array<uint8_t, 9> fixture = { 0u, 1u, 10u, 11u, 12u, 32u, 64u, 128u, 255u };
    size_t gamma22Mutations = 0;
    for (const uint8_t code : fixture)
    {
        const double encoded = static_cast<double>(code) / 255.0;
        const double expected = encoded <= 0.04045 ? encoded / 12.92 : std::pow((encoded + 0.055) / 1.055, 2.4);
        CHECK(oka::projector::srgb8ToLinear(code) == doctest::Approx(expected).epsilon(2e-6).scale(1e-8));

        const double gamma22 = std::pow(encoded, 2.2);
        if (std::abs(gamma22 - expected) > 1e-5)
        {
            ++gamma22Mutations;
        }
    }
    // Mutation: stb's default LDR-to-float power curve disagrees at every
    // useful midtone; endpoints alone would not make this test sensitive.
    CHECK(gamma22Mutations >= 5u);
}

TEST_CASE("projector HDR sanitization preserves finite positive radiance")
{
    std::array<float, 8> pixels = {
        4.0f, -1.0f, std::numeric_limits<float>::infinity(), 2.0f, std::numeric_limits<float>::quiet_NaN(),
        0.5f, 1.0f,  std::numeric_limits<float>::quiet_NaN()
    };
    oka::projector::sanitizeLinearRgba(pixels.data(), 2u);
    CHECK(pixels[0] == 4.0f);
    CHECK(pixels[1] == 0.0f);
    CHECK(pixels[2] == 0.0f);
    CHECK(pixels[3] == 1.0f);
    CHECK(pixels[4] == 0.0f);
    CHECK(pixels[5] == 0.5f);
    CHECK(pixels[6] == 1.0f);
    CHECK(pixels[7] == 1.0f);
}

TEST_CASE("Metal projector images bypass the material UNORM and compression path")
{
    const std::filesystem::path repository = std::filesystem::path(STRELKA_TEST_ASSETS_DIR).parent_path().parent_path();
    std::ifstream sourceFile(repository / "src/render/metal/MetalLights.mm");
    REQUIRE(sourceFile.good());
    const std::string source((std::istreambuf_iterator<char>(sourceFile)), std::istreambuf_iterator<char>());
    const size_t begin = source.find("void MetalLights::loadProjectorImages");
    const size_t end = source.find("void MetalLights::upload", begin);
    REQUIRE(begin != std::string::npos);
    REQUIRE(end != std::string::npos);
    const std::string body = source.substr(begin, end - begin);

    CHECK(body.find("loadProjectorFromFile") != std::string::npos);
    CHECK(body.find("loadFromFile(") == std::string::npos);

    // Mutation: the old RGBA8 path clamps authored HDR emission before the
    // shader can evaluate it. A linear 4.0 texel must therefore use float
    // storage rather than merely toggling the sRGB bit on the old upload.
    CHECK(std::clamp(4.0f, 0.0f, 1.0f) != doctest::Approx(4.0f));
}

TEST_CASE("OptiX projector texture lookup is explicitly bounds checked")
{
    const std::filesystem::path repository = std::filesystem::path(STRELKA_TEST_ASSETS_DIR).parent_path().parent_path();
    std::ifstream paramsFile(repository / "src/render/optix/OptixRenderParams.h");
    std::ifstream shaderFile(repository / "src/shaders/optix/OptixRender_closest_hit.cu");
    REQUIRE(paramsFile.good());
    REQUIRE(shaderFile.good());
    const std::string params((std::istreambuf_iterator<char>(paramsFile)), std::istreambuf_iterator<char>());
    const std::string shader((std::istreambuf_iterator<char>(shaderFile)), std::istreambuf_iterator<char>());
    const size_t begin = shader.find("static __forceinline__ __device__ float3 projectorEmission");
    const size_t end = shader.find("static __forceinline__ __device__ float3 emittedLightRadiance", begin);
    REQUIRE(begin != std::string::npos);
    REQUIRE(end != std::string::npos);
    const std::string body = shader.substr(begin, end - begin);

    CHECK(params.find("numProjectorTextures") != std::string::npos);
    CHECK(body.find("params.scene.numProjectorTextures") != std::string::npos);
}

#if defined(__APPLE__)
TEST_CASE("Metal projector loader preserves HDR radiance on an actual device")
{
    NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();
    MTL::Device* device = MTL::CreateSystemDefaultDevice();
    REQUIRE(device != nullptr);

    const std::filesystem::path path =
        std::filesystem::temp_directory_path() / ("strelka-projector-" + std::to_string(getpid()) + ".hdr");
    {
        std::ofstream output(path, std::ios::binary);
        REQUIRE(output.good());
        output << "#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y 1 +X 1\n";
        const std::array<char, 4> rgbe = { static_cast<char>(128), static_cast<char>(64), static_cast<char>(32),
                                           static_cast<char>(131) };
        output.write(rgbe.data(), static_cast<std::streamsize>(rgbe.size()));
    }

    oka::metal::MetalTextures textures;
    textures.init(device, nullptr, nullptr);
    MTL::Texture* texture = textures.loadProjectorFromFile(path.string());
    REQUIRE(texture != nullptr);
    CHECK(texture->pixelFormat() == MTL::PixelFormatRGBA32Float);
    CHECK(texture->mipmapLevelCount() == 1u);

    std::array<float, 4> pixel{};
    texture->getBytes(pixel.data(), 4u * sizeof(float), MTL::Region::Make2D(0, 0, 1, 1), 0u);
    CHECK(pixel[0] == doctest::Approx(4.0f));
    CHECK(pixel[1] == doctest::Approx(2.0f));
    CHECK(pixel[2] == doctest::Approx(1.0f));
    CHECK(pixel[3] == doctest::Approx(1.0f));

    texture->release();
    std::error_code error;
    std::filesystem::remove(path, error);
    pool->release();
}
#endif
