#include <doctest/doctest.h>

#include <shading/texture_transform.h>

#include <cmath>

// ---------------------------------------------------------------------------
// KHR_texture_transform, COLOR_0 and glTF alpha modes, on the OptiX side.
//
// The composition order is the thing worth pinning. Scale before rotation and
// translation last is what the spec's row-vector form says, and getting it wrong
// is invisible at rotation 0 -- which is what every exporter writes by default --
// and wrong everywhere else. A render test cannot tell the two apart on any scene
// in the ladder.
// ---------------------------------------------------------------------------

namespace
{

MaterialParams identity_transform()
{
    MaterialParams m = {};
    m.uv_scale_x = 1.0f;
    m.uv_scale_y = 1.0f;
    m.uv_offset_x = 0.0f;
    m.uv_offset_y = 0.0f;
    m.uv_rotation = 0.0f;
    return m;
}

bool near(float a, float b, float eps = 1e-5f)
{
    return std::fabs(a - b) < eps;
}

} // namespace

TEST_CASE("the identity transform leaves uv alone")
{
    const MaterialParams m = identity_transform();
    const float2 uv = apply_texture_transform(make_float2(0.3f, 0.7f), m);
    CHECK(near(uv.x, 0.3f));
    CHECK(near(uv.y, 0.7f));
}

TEST_CASE("a zero-initialised material is not a collapsed texture")
{
    // The OptiX backend builds a fallback material by value-initialising
    // MaterialParams, which leaves the scale at zero. Read literally that maps
    // every uv onto the offset -- one texel smeared over the whole surface, which
    // looks like a texture that failed to load rather than like a missing
    // extension.
    const MaterialParams m = {};
    const float2 uv = apply_texture_transform(make_float2(0.25f, 0.75f), m);
    CHECK(near(uv.x, 0.25f));
    CHECK(near(uv.y, 0.75f));
}

TEST_CASE("scale applies before rotation, and translation last")
{
    MaterialParams m = identity_transform();
    m.uv_scale_x = 2.0f;
    m.uv_scale_y = 3.0f;
    m.uv_rotation = 1.5707963268f; // 90 degrees
    m.uv_offset_x = 0.1f;
    m.uv_offset_y = -0.2f;

    // [u v] -> (u*sx*cos - v*sy*sin, u*sx*sin + v*sy*cos) + t
    // At 90 degrees: (-v*sy, u*sx) + t.
    const float2 uv = apply_texture_transform(make_float2(1.0f, 1.0f), m);
    CHECK(near(uv.x, -3.0f + 0.1f, 1e-4f));
    CHECK(near(uv.y, 2.0f - 0.2f, 1e-4f));

    // The order matters: rotating first and then scaling would give
    // (-1*2, 1*3) = (-2, 3) instead. Assert the difference explicitly so a
    // refactor that swaps them cannot pass.
    CHECK_FALSE(near(uv.x, -2.0f + 0.1f, 1e-3f));
}

TEST_CASE("tiling is a scale, and the offset does not scale with it")
{
    // The marble worktop in the bathroom repeats 2x2 with no offset; the plant's
    // ramp 50x50. Both depend on the translation staying in transformed space.
    MaterialParams m = identity_transform();
    m.uv_scale_x = 2.0f;
    m.uv_scale_y = 2.0f;
    m.uv_offset_x = 0.5f;
    const float2 uv = apply_texture_transform(make_float2(0.5f, 0.5f), m);
    CHECK(near(uv.x, 1.5f));
    CHECK(near(uv.y, 1.0f));
}

TEST_CASE("COLOR_0 unpacks as linear RGBA8, low byte first")
{
    CHECK(near(unpack_vertex_color(0xFFFFFFFFu).x, 1.0f));
    CHECK(near(unpack_vertex_color(0xFFFFFFFFu).y, 1.0f));
    CHECK(near(unpack_vertex_color(0xFFFFFFFFu).z, 1.0f));

    // Pure red is 0x000000FF, not 0xFF000000: the packer writes r in the low
    // byte. Reading it the other way tints every vertex-coloured mesh blue.
    const float3 red = unpack_vertex_color(0x000000FFu);
    CHECK(near(red.x, 1.0f));
    CHECK(near(red.y, 0.0f));
    CHECK(near(red.z, 0.0f));

    // And no transfer function: 128/255 stays 0.502, it does not decode to 0.216.
    const float3 grey = unpack_vertex_color(0x00808080u);
    CHECK(near(grey.x, 128.0f / 255.0f, 1e-4f));
}

TEST_CASE("alpha modes resolve to a coverage the renderer can use blindly")
{
    MaterialParams m = identity_transform();

    m.alpha_mode = ALPHA_MODE_OPAQUE;
    CHECK(near(resolve_opacity(m, 0.25f), 1.0f));

    m.alpha_mode = ALPHA_MODE_BLEND;
    CHECK(near(resolve_opacity(m, 0.25f), 0.25f));
    CHECK(near(resolve_opacity(m, 1.5f), 1.0f));
    CHECK(near(resolve_opacity(m, -0.5f), 0.0f));

    m.alpha_mode = ALPHA_MODE_MASK;
    m.alpha_cutoff = 0.5f;
    CHECK(near(resolve_opacity(m, 0.49f), 0.0f));
    CHECK(near(resolve_opacity(m, 0.5f), 1.0f));
    CHECK(near(resolve_opacity(m, 0.9f), 1.0f));
}
