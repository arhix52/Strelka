// ============================================================================
// test_materialx_inputs.cpp
//
// How the MaterialX front-end resolves a shader input: to a folded constant, to
// a texture, and with whatever the document says about that texture's encoding.
//
// The loader resolves each shader input to a constant or to an image. Real
// libraries rarely state a constant outright: they scale a tint, mix two
// colours, remap a roughness. Before folding, every one of those landed in
// `unsupported` and the parameter silently kept its specification default --
// a material that ignored the document while reporting success, which is the
// failure mode these cases exist to refuse.
//
// Written against documents small enough to read, because the arithmetic is the
// thing under test and a production asset would hide it.
// ============================================================================

#include <doctest/doctest.h>

#include <strelka/material/openpbr/openpbr_params.h>
#include <strelka/sceneloader/materialx_loader.h>

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <set>
#include <string>

namespace
{

// Writes a document whose only material is `M_test`, with `body` between the
// shader node and the material, and returns the path.
std::filesystem::path writeDoc(const char* stem, const std::string& shaderInputs, const std::string& body)
{
    const std::filesystem::path path =
        std::filesystem::temp_directory_path() / (std::string("strelka_fold_") + stem + ".mtlx");
    std::ofstream out(path);
    out << R"(<?xml version="1.0"?>)"
        << "\n<materialx version=\"1.39\">\n"
        << "  <open_pbr_surface name=\"SR\" type=\"surfaceshader\">\n"
        << shaderInputs << "  </open_pbr_surface>\n"
        << body << "  <surfacematerial name=\"M_test\" type=\"material\">\n"
        << "    <input name=\"surfaceshader\" type=\"surfaceshader\" nodename=\"SR\"/>\n"
        << "  </surfacematerial>\n</materialx>\n";
    return path;
}

oka::mtlx::MaterialXMaterial load(const std::filesystem::path& path)
{
    const oka::mtlx::MaterialXDocumentData doc = oka::mtlx::loadMaterialXDocument(path.string());
    REQUIRE(doc.materials.size() == 1);
    return doc.materials.front();
}

} // namespace

TEST_CASE("a tint scaled by a float folds into base_color")
{
    const auto path = writeDoc("multiply",
                               "    <input name=\"base_color\" type=\"color3\" nodename=\"tint\"/>\n",
                               "  <multiply name=\"tint\" type=\"color3\">\n"
                               "    <input name=\"in1\" type=\"color3\" value=\"0.8, 0.4, 0.2\"/>\n"
                               "    <input name=\"in2\" type=\"float\" value=\"0.5\"/>\n"
                               "  </multiply>\n");
    const oka::mtlx::MaterialXMaterial m = load(path);

    CHECK(m.params.base_color.r == doctest::Approx(0.4f));
    CHECK(m.params.base_color.g == doctest::Approx(0.2f));
    CHECK(m.params.base_color.b == doctest::Approx(0.1f));
    CHECK(m.unsupported.empty());
    std::filesystem::remove(path);
}

TEST_CASE("mix reads bg at 0 and fg at 1, which is the stdlib's way round")
{
    const auto path = writeDoc("mix",
                               "    <input name=\"base_color\" type=\"color3\" nodename=\"blend\"/>\n",
                               "  <mix name=\"blend\" type=\"color3\">\n"
                               "    <input name=\"fg\" type=\"color3\" value=\"1, 0, 0\"/>\n"
                               "    <input name=\"bg\" type=\"color3\" value=\"0, 0, 1\"/>\n"
                               "    <input name=\"mix\" type=\"float\" value=\"0.25\"/>\n"
                               "  </mix>\n");
    const oka::mtlx::MaterialXMaterial m = load(path);

    CHECK(m.params.base_color.r == doctest::Approx(0.25f));
    CHECK(m.params.base_color.b == doctest::Approx(0.75f));
    std::filesystem::remove(path);
}

TEST_CASE("nested arithmetic folds all the way down")
{
    // clamp(remap(0.5, 0..1 -> 0.2..1.0), 0, 0.6) = clamp(0.6, 0, 0.6) = 0.6
    const auto path = writeDoc("nested",
                               "    <input name=\"specular_roughness\" type=\"float\" nodename=\"rough\"/>\n",
                               "  <remap name=\"mapped\" type=\"float\">\n"
                               "    <input name=\"in\" type=\"float\" value=\"0.5\"/>\n"
                               "    <input name=\"outlow\" type=\"float\" value=\"0.2\"/>\n"
                               "    <input name=\"outhigh\" type=\"float\" value=\"1.0\"/>\n"
                               "  </remap>\n"
                               "  <clamp name=\"rough\" type=\"float\">\n"
                               "    <input name=\"in\" type=\"float\" nodename=\"mapped\"/>\n"
                               "    <input name=\"low\" type=\"float\" value=\"0\"/>\n"
                               "    <input name=\"high\" type=\"float\" value=\"0.6\"/>\n"
                               "  </clamp>\n");
    const oka::mtlx::MaterialXMaterial m = load(path);

    CHECK(m.params.specular_roughness == doctest::Approx(0.6f));
    CHECK(m.unsupported.empty());
    std::filesystem::remove(path);
}

TEST_CASE("combine3 and extract move components around")
{
    const auto path = writeDoc("combine",
                               "    <input name=\"base_color\" type=\"color3\" nodename=\"rgb\"/>\n"
                               "    <input name=\"specular_roughness\" type=\"float\" nodename=\"pick\"/>\n",
                               "  <combine3 name=\"rgb\" type=\"color3\">\n"
                               "    <input name=\"in1\" type=\"float\" value=\"0.1\"/>\n"
                               "    <input name=\"in2\" type=\"float\" value=\"0.2\"/>\n"
                               "    <input name=\"in3\" type=\"float\" value=\"0.3\"/>\n"
                               "  </combine3>\n"
                               "  <extract name=\"pick\" type=\"float\">\n"
                               "    <input name=\"in\" type=\"color3\" nodename=\"rgb\"/>\n"
                               "    <input name=\"index\" type=\"integer\" value=\"1\"/>\n"
                               "  </extract>\n");
    const oka::mtlx::MaterialXMaterial m = load(path);

    CHECK(m.params.base_color.g == doctest::Approx(0.2f));
    CHECK(m.params.specular_roughness == doctest::Approx(0.2f));
    std::filesystem::remove(path);
}

TEST_CASE("a scalar broadcasts across a colour operand")
{
    const auto path = writeDoc("broadcast",
                               "    <input name=\"base_color\" type=\"color3\" nodename=\"sum\"/>\n",
                               "  <add name=\"sum\" type=\"color3\">\n"
                               "    <input name=\"in1\" type=\"color3\" value=\"0.1, 0.2, 0.3\"/>\n"
                               "    <input name=\"in2\" type=\"float\" value=\"0.5\"/>\n"
                               "  </add>\n");
    const oka::mtlx::MaterialXMaterial m = load(path);

    CHECK(m.params.base_color.r == doctest::Approx(0.6f));
    CHECK(m.params.base_color.g == doctest::Approx(0.7f));
    CHECK(m.params.base_color.b == doctest::Approx(0.8f));
    std::filesystem::remove(path);
}

// The line the fold must not cross. A texture has no value until the pixel is
// shaded, so an expression containing one cannot become a number in
// OpenPBRParams -- and half-folding it, by taking the constant operand and
// dropping the image, would be worse than saying so.
TEST_CASE("an expression containing an image does not fold, and says which node")
{
    const auto path = writeDoc("withimage",
                               "    <input name=\"base_color\" type=\"color3\" nodename=\"tinted\"/>\n",
                               "  <image name=\"tex\" type=\"color3\">\n"
                               "    <input name=\"file\" type=\"filename\" value=\"albedo.png\"/>\n"
                               "  </image>\n"
                               "  <multiply name=\"tinted\" type=\"color3\">\n"
                               "    <input name=\"in1\" type=\"color3\" nodename=\"tex\"/>\n"
                               "    <input name=\"in2\" type=\"float\" value=\"0.5\"/>\n"
                               "  </multiply>\n");
    const oka::mtlx::MaterialXMaterial m = load(path);

    CHECK(m.texPaths[OPENPBR_TEX_BASE_COLOR].empty());
    REQUIRE(m.unsupported.size() == 1);
    CHECK(m.unsupported.front().find("multiply") != std::string::npos);
    std::filesystem::remove(path);
}

TEST_CASE("a bare image still reaches its slot")
{
    const auto path = writeDoc("bareimage",
                               "    <input name=\"base_color\" type=\"color3\" nodename=\"tex\"/>\n",
                               "  <image name=\"tex\" type=\"color3\">\n"
                               "    <input name=\"file\" type=\"filename\" value=\"albedo.png\"/>\n"
                               "  </image>\n");
    const oka::mtlx::MaterialXMaterial m = load(path);

    CHECK(m.texPaths[OPENPBR_TEX_BASE_COLOR].find("albedo.png") != std::string::npos);
    CHECK(m.unsupported.empty());
    std::filesystem::remove(path);
}

// The library MaterialX ships is the honest measure of what folding bought: 48
// materials across 50 documents, and the only inputs left unexpressed are the
// ones no amount of arithmetic could reduce to a number.
TEST_CASE("only the procedural examples are left unexpressed")
{
    const std::filesystem::path root = std::filesystem::path(STRELKA_MATERIALX_ROOT) / "resources/Materials/Examples";
    REQUIRE(std::filesystem::exists(root));

    std::set<std::string> documentsWithGaps;
    int materials = 0;
    for (const auto& entry : std::filesystem::recursive_directory_iterator(root))
    {
        if (entry.path().extension() != ".mtlx")
        {
            continue;
        }
        const oka::mtlx::MaterialXDocumentData doc = oka::mtlx::loadMaterialXDocument(entry.path().string());
        for (const oka::mtlx::MaterialXMaterial& m : doc.materials)
        {
            ++materials;
            if (!m.unsupported.empty())
            {
                documentsWithGaps.insert(entry.path().stem().string());
            }
        }
    }

    CHECK(materials >= 48);
    // Two procedural patterns and one hex-tiled image. A noise function has no
    // constant to fold to and a hextiledimage is a texture node this loader does
    // not read; everything else in the shipped library now binds completely.
    const std::set<std::string> expected{ "standard_surface_marble_solid", "standard_surface_brick_procedural",
                                          "standard_surface_onyx_hextiled" };
    CHECK(documentsWithGaps == expected);
}

// ----------------------------------------------------------------------------
// Colour spaces.
//
// The renderer guesses a slot's encoding from what the slot is for -- base
// colour sRGB, roughness linear -- which is glTF's convention and what the
// chess set happens to agree with. Only the document can say when an asset
// departs from it, and until this was read, it could not.
// ----------------------------------------------------------------------------

TEST_CASE("a stated colorspace overrides the slot's guess in both directions")
{
    const auto path = writeDoc("colorspace",
                               "    <input name=\"base_color\" type=\"color3\" nodename=\"albedo\"/>\n"
                               "    <input name=\"specular_roughness\" type=\"float\" nodename=\"rough\"/>\n"
                               "    <input name=\"base_metalness\" type=\"float\" nodename=\"metal\"/>\n",
                               "  <image name=\"albedo\" type=\"color3\">\n"
                               "    <input name=\"file\" type=\"filename\" value=\"a.png\" colorspace=\"lin_rec709\"/>\n"
                               "  </image>\n"
                               "  <image name=\"rough\" type=\"float\">\n"
                               "    <input name=\"file\" type=\"filename\" value=\"r.png\" colorspace=\"srgb_texture\"/>\n"
                               "  </image>\n"
                               "  <image name=\"metal\" type=\"float\">\n"
                               "    <input name=\"file\" type=\"filename\" value=\"m.png\"/>\n"
                               "  </image>\n");
    const oka::mtlx::MaterialXMaterial m = load(path);

    // A base colour the document calls linear must not be decoded again.
    CHECK(m.texColorSpace[OPENPBR_TEX_BASE_COLOR] == oka::TexColorSpace::Linear);
    // A roughness authored through a gamma curve must be.
    CHECK(m.texColorSpace[OPENPBR_TEX_SPECULAR_ROUGHNESS] == oka::TexColorSpace::Srgb);
    // Silence leaves the slot's own guess in charge, which is what keeps every
    // document that says nothing rendering exactly as it did.
    CHECK(m.texColorSpace[OPENPBR_TEX_BASE_METALNESS] == oka::TexColorSpace::Unspecified);
    std::filesystem::remove(path);
}

TEST_CASE("colorspace names are read by convention, and the gamut is not claimed")
{
    // acescg is linear light with different primaries. Honouring the curve and
    // ignoring the gamut is the truthful half of the answer; claiming the
    // gamut would be the untruthful one.
    const auto path = writeDoc("acescg",
                               "    <input name=\"base_color\" type=\"color3\" nodename=\"albedo\"/>\n",
                               "  <image name=\"albedo\" type=\"color3\">\n"
                               "    <input name=\"file\" type=\"filename\" value=\"a.exr\" colorspace=\"acescg\"/>\n"
                               "  </image>\n");
    const oka::mtlx::MaterialXMaterial m = load(path);
    CHECK(m.texColorSpace[OPENPBR_TEX_BASE_COLOR] == oka::TexColorSpace::Linear);
    std::filesystem::remove(path);
}

TEST_CASE("an unrecognised colorspace leaves the slot alone rather than guessing")
{
    const auto path = writeDoc("weirdspace",
                               "    <input name=\"base_color\" type=\"color3\" nodename=\"albedo\"/>\n",
                               "  <image name=\"albedo\" type=\"color3\">\n"
                               "    <input name=\"file\" type=\"filename\" value=\"a.png\" "
                               "colorspace=\"studio_private_log\"/>\n"
                               "  </image>\n");
    const oka::mtlx::MaterialXMaterial m = load(path);
    CHECK(m.texColorSpace[OPENPBR_TEX_BASE_COLOR] == oka::TexColorSpace::Unspecified);
    CHECK(!m.texPaths[OPENPBR_TEX_BASE_COLOR].empty());
    std::filesystem::remove(path);
}

TEST_CASE("the chess set's own tagging agrees with the slot defaults")
{
    // Which is why honouring the document changed none of its pixels: it tags
    // base colours srgb_texture and leaves metalness, roughness and normals
    // bare. A regression here would move an asset this tree renders daily.
    const oka::mtlx::MaterialXDocumentData doc = oka::mtlx::loadMaterialXDocument(
        std::string(STRELKA_MATERIALX_ROOT) +
        "/resources/Materials/Examples/StandardSurface/standard_surface_chess_set.mtlx");
    REQUIRE(!doc.materials.empty());

    std::set<size_t> tagged;
    for (const oka::mtlx::MaterialXMaterial& m : doc.materials)
    {
        for (size_t slot = 0; slot < MAX_OPENPBR_TEXTURES; ++slot)
        {
            if (m.texColorSpace[slot] == oka::TexColorSpace::Srgb)
            {
                tagged.insert(slot);
            }
            // Nothing here is called linear, so nothing here can turn a slot
            // the renderer decodes off -- the only way this document could
            // have moved a pixel.
            CHECK(m.texColorSpace[slot] != oka::TexColorSpace::Linear);
        }
    }

    // Exactly the three colour slots openpbrSlotKind() already decodes as sRGB.
    // Their agreement is why honouring the document changed none of its pixels,
    // and a fourth slot appearing here would mean it now does.
    const std::set<size_t> expected{ OPENPBR_TEX_BASE_COLOR, OPENPBR_TEX_SUBSURFACE_COLOR,
                                     OPENPBR_TEX_SUBSURFACE_RADIUS };
    CHECK(tagged == expected);
}

// ----------------------------------------------------------------------------
// UV placement.
//
// Transcribed from the stdlib nodegraphs, because every field here means the
// opposite of what its name suggests: place2d divides by scale and subtracts
// offset, tiledimage subtracts after multiplying, and MaterialX's rotate2d
// turns clockwise where the shader's turns the other way. A texture placed
// almost right is the hardest kind of wrong to see, so these pin the algebra
// rather than the plumbing.
// ----------------------------------------------------------------------------

TEST_CASE("place2d scale divides, and reaches the shader as its reciprocal")
{
    const auto path = writeDoc("place_scale",
                               "    <input name=\"base_color\" type=\"color3\" nodename=\"albedo\"/>\n",
                               "  <place2d name=\"pl\" type=\"vector2\">\n"
                               "    <input name=\"scale\" type=\"vector2\" value=\"2, 4\"/>\n"
                               "  </place2d>\n"
                               "  <image name=\"albedo\" type=\"color3\">\n"
                               "    <input name=\"file\" type=\"filename\" value=\"a.png\"/>\n"
                               "    <input name=\"texcoord\" type=\"vector2\" nodename=\"pl\"/>\n"
                               "  </image>\n");
    const oka::mtlx::MaterialXMaterial m = load(path);

    CHECK(m.params.uv_scale_x == doctest::Approx(0.5f));
    CHECK(m.params.uv_scale_y == doctest::Approx(0.25f));
    CHECK(m.params.uv_offset_x == doctest::Approx(0.0f));
    CHECK(m.unsupported.empty());
    std::filesystem::remove(path);
}

TEST_CASE("place2d offset subtracts")
{
    const auto path = writeDoc("place_offset",
                               "    <input name=\"base_color\" type=\"color3\" nodename=\"albedo\"/>\n",
                               "  <place2d name=\"pl\" type=\"vector2\">\n"
                               "    <input name=\"offset\" type=\"vector2\" value=\"0.25, 0.5\"/>\n"
                               "  </place2d>\n"
                               "  <image name=\"albedo\" type=\"color3\">\n"
                               "    <input name=\"file\" type=\"filename\" value=\"a.png\"/>\n"
                               "    <input name=\"texcoord\" type=\"vector2\" nodename=\"pl\"/>\n"
                               "  </image>\n");
    const oka::mtlx::MaterialXMaterial m = load(path);

    CHECK(m.params.uv_offset_x == doctest::Approx(-0.25f));
    CHECK(m.params.uv_offset_y == doctest::Approx(-0.5f));
    std::filesystem::remove(path);
}

TEST_CASE("place2d rotation arrives negated, because the two conventions differ")
{
    const auto path = writeDoc("place_rotate",
                               "    <input name=\"base_color\" type=\"color3\" nodename=\"albedo\"/>\n",
                               "  <place2d name=\"pl\" type=\"vector2\">\n"
                               "    <input name=\"rotate\" type=\"float\" value=\"90\"/>\n"
                               "  </place2d>\n"
                               "  <image name=\"albedo\" type=\"color3\">\n"
                               "    <input name=\"file\" type=\"filename\" value=\"a.png\"/>\n"
                               "    <input name=\"texcoord\" type=\"vector2\" nodename=\"pl\"/>\n"
                               "  </image>\n");
    const oka::mtlx::MaterialXMaterial m = load(path);

    // MaterialX rotate2d is (ca*x + sa*y, -sa*x + ca*y); the shader's is the
    // other way round, so 90 degrees arrives as -pi/2.
    CHECK(m.params.uv_rotation == doctest::Approx(-1.57079633f));
    std::filesystem::remove(path);
}

TEST_CASE("a pivoted rotation leaves the pivot where it was")
{
    // The property the pivot terms exist for: whatever the rotation and scale,
    // the pivot itself must map to itself. Checked by running the shader's own
    // transform on it.
    const auto path = writeDoc("place_pivot",
                               "    <input name=\"base_color\" type=\"color3\" nodename=\"albedo\"/>\n",
                               "  <place2d name=\"pl\" type=\"vector2\">\n"
                               "    <input name=\"pivot\" type=\"vector2\" value=\"0.5, 0.5\"/>\n"
                               "    <input name=\"rotate\" type=\"float\" value=\"37\"/>\n"
                               "    <input name=\"scale\" type=\"vector2\" value=\"2, 2\"/>\n"
                               "  </place2d>\n"
                               "  <image name=\"albedo\" type=\"color3\">\n"
                               "    <input name=\"file\" type=\"filename\" value=\"a.png\"/>\n"
                               "    <input name=\"texcoord\" type=\"vector2\" nodename=\"pl\"/>\n"
                               "  </image>\n");
    const oka::mtlx::MaterialXMaterial m = load(path);

    const float c = std::cos(m.params.uv_rotation);
    const float s = std::sin(m.params.uv_rotation);
    const float px = 0.5f * m.params.uv_scale_x;
    const float py = 0.5f * m.params.uv_scale_y;
    const float outX = px * c - py * s + m.params.uv_offset_x;
    const float outY = px * s + py * c + m.params.uv_offset_y;

    CHECK(outX == doctest::Approx(0.5f).epsilon(1e-5));
    CHECK(outY == doctest::Approx(0.5f).epsilon(1e-5));
    std::filesystem::remove(path);
}

TEST_CASE("tiledimage tiling multiplies and its offset subtracts")
{
    const auto path = writeDoc("tiled",
                               "    <input name=\"base_color\" type=\"color3\" nodename=\"albedo\"/>\n",
                               "  <tiledimage name=\"albedo\" type=\"color3\">\n"
                               "    <input name=\"file\" type=\"filename\" value=\"a.png\"/>\n"
                               "    <input name=\"uvtiling\" type=\"vector2\" value=\"3, 5\"/>\n"
                               "    <input name=\"uvoffset\" type=\"vector2\" value=\"0.1, 0.2\"/>\n"
                               "  </tiledimage>\n");
    const oka::mtlx::MaterialXMaterial m = load(path);

    CHECK(m.params.uv_scale_x == doctest::Approx(3.0f));
    CHECK(m.params.uv_scale_y == doctest::Approx(5.0f));
    CHECK(m.params.uv_offset_x == doctest::Approx(-0.1f));
    CHECK(m.params.uv_offset_y == doctest::Approx(-0.2f));
    std::filesystem::remove(path);
}

TEST_CASE("real-world sizing is named rather than applied as one")
{
    // It needs the scene's unit system, which the renderer does not carry.
    const auto path = writeDoc("realworld",
                               "    <input name=\"base_color\" type=\"color3\" nodename=\"albedo\"/>\n",
                               "  <tiledimage name=\"albedo\" type=\"color3\">\n"
                               "    <input name=\"file\" type=\"filename\" value=\"a.png\"/>\n"
                               "    <input name=\"realworldtilesize\" type=\"vector2\" value=\"0.5, 0.5\"/>\n"
                               "  </tiledimage>\n");
    const oka::mtlx::MaterialXMaterial m = load(path);

    REQUIRE(m.unsupported.size() == 1);
    CHECK(m.unsupported.front().find("realworldtilesize") != std::string::npos);
    std::filesystem::remove(path);
}

TEST_CASE("two maps placed differently: the second is reported, not silently dropped")
{
    const auto path = writeDoc("twoplacements",
                               "    <input name=\"base_color\" type=\"color3\" nodename=\"albedo\"/>\n"
                               "    <input name=\"specular_roughness\" type=\"float\" nodename=\"rough\"/>\n",
                               "  <place2d name=\"p1\" type=\"vector2\">\n"
                               "    <input name=\"scale\" type=\"vector2\" value=\"2, 2\"/>\n"
                               "  </place2d>\n"
                               "  <place2d name=\"p2\" type=\"vector2\">\n"
                               "    <input name=\"scale\" type=\"vector2\" value=\"8, 8\"/>\n"
                               "  </place2d>\n"
                               "  <image name=\"albedo\" type=\"color3\">\n"
                               "    <input name=\"file\" type=\"filename\" value=\"a.png\"/>\n"
                               "    <input name=\"texcoord\" type=\"vector2\" nodename=\"p1\"/>\n"
                               "  </image>\n"
                               "  <image name=\"rough\" type=\"float\">\n"
                               "    <input name=\"file\" type=\"filename\" value=\"r.png\"/>\n"
                               "    <input name=\"texcoord\" type=\"vector2\" nodename=\"p2\"/>\n"
                               "  </image>\n");
    const oka::mtlx::MaterialXMaterial m = load(path);

    // One transform per material, so the first wins and the loss is stated.
    CHECK(m.params.uv_scale_x == doctest::Approx(0.5f));
    REQUIRE(m.unsupported.size() == 1);
    CHECK(m.unsupported.front().find("second placement") != std::string::npos);
    std::filesystem::remove(path);
}
