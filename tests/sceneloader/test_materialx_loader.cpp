// ============================================================================
// test_materialx_loader.cpp
//
// Reads the Open Chess Set's own MaterialX document -- the one shipped in the
// MaterialX repository, not a fixture written to pass -- and checks that every
// input it drives from a nodegraph arrives in the slot the renderer will look
// for it in.
//
// That last part is the whole point. A texture that lands in no slot is not an
// error anywhere: the loader warns, the material still renders, and the missing
// map reads as "this material is a bit flat" rather than as a defect. The chess
// set is a good witness because it drives seven different inputs from images
// across fifteen materials, including two -- subsurface weight and subsurface
// radius -- that only got slots because this asset asked for them.
// ============================================================================

#include <doctest/doctest.h>

#include <strelka/material/openpbr/openpbr_params.h>
#include <strelka/sceneloader/materialx_loader.h>

#include <algorithm>
#include <filesystem>
#include <string>

namespace
{
std::string chessDocument()
{
    return std::string(STRELKA_MATERIALX_ROOT) +
           "/resources/Materials/Examples/StandardSurface/standard_surface_chess_set.mtlx";
}

const oka::mtlx::MaterialXMaterial* find(const std::vector<oka::mtlx::MaterialXMaterial>& all, const char* name)
{
    const auto it = std::ranges::find_if(all, [&](const oka::mtlx::MaterialXMaterial& m) { return m.name == name; });
    return it == all.end() ? nullptr : &*it;
}
} // namespace

TEST_CASE("the Open Chess Set parses into fifteen materials and a look")
{
    REQUIRE(std::filesystem::exists(chessDocument()));
    const oka::mtlx::MaterialXDocumentData doc = oka::mtlx::loadMaterialXDocument(chessDocument());

    CHECK(doc.materials.size() == 15);
    // Its glTF carries two placeholder materials for fifteen pieces, so the look
    // is the only thing that says which piece is which.
    CHECK(doc.assignments.size() == 15);
}

TEST_CASE("every image the chess set drives lands in a slot")
{
    const oka::mtlx::MaterialXDocumentData doc = oka::mtlx::loadMaterialXDocument(chessDocument());
    REQUIRE(!doc.materials.empty());

    SUBCASE("a king: base colour, metalness, roughness, normal, and both subsurface maps")
    {
        const oka::mtlx::MaterialXMaterial* king = find(doc.materials, "M_King_B");
        REQUIRE(king != nullptr);

        CHECK_FALSE(king->texPaths[OPENPBR_TEX_BASE_COLOR].empty());
        CHECK_FALSE(king->texPaths[OPENPBR_TEX_BASE_METALNESS].empty());
        CHECK_FALSE(king->texPaths[OPENPBR_TEX_SPECULAR_ROUGHNESS].empty());
        CHECK_FALSE(king->texPaths[OPENPBR_TEX_GEOMETRY_NORMAL].empty());

        // The two the slot table grew for. Without them the piece renders as an
        // opaque marble that never enters its own medium, and nothing says so.
        CHECK_FALSE(king->texPaths[OPENPBR_TEX_SUBSURFACE_WEIGHT].empty());
        CHECK_FALSE(king->texPaths[OPENPBR_TEX_SUBSURFACE_COLOR].empty());
        CHECK_FALSE(king->texPaths[OPENPBR_TEX_SUBSURFACE_RADIUS].empty());

        // Resolved against the document, so the renderer can open them.
        CHECK(king->texPaths[OPENPBR_TEX_BASE_COLOR].find("chess_set/") != std::string::npos);
        CHECK(std::filesystem::exists(king->texPaths[OPENPBR_TEX_BASE_COLOR]));
        CHECK(std::filesystem::exists(king->texPaths[OPENPBR_TEX_SUBSURFACE_WEIGHT]));
    }

    SUBCASE("a pawn top: glass, and only the two maps its graph exposes")
    {
        const oka::mtlx::MaterialXMaterial* pawn = find(doc.materials, "M_Pawn_Top_W");
        REQUIRE(pawn != nullptr);

        CHECK_FALSE(pawn->texPaths[OPENPBR_TEX_SPECULAR_ROUGHNESS].empty());
        CHECK_FALSE(pawn->texPaths[OPENPBR_TEX_GEOMETRY_NORMAL].empty());
        CHECK(pawn->texPaths[OPENPBR_TEX_BASE_COLOR].empty());

        // standard_surface's transmission maps onto the OpenPBR weight, and the
        // tint with it: this is what makes the pawn tops amber rather than white.
        CHECK(pawn->params.transmission_weight == doctest::Approx(1.0f));
        CHECK(pawn->params.transmission_color.b < pawn->params.transmission_color.r);
    }

    SUBCASE("nothing in the document was left unexpressed")
    {
        for (const oka::mtlx::MaterialXMaterial& m : doc.materials)
        {
            CAPTURE(m.name);
            CHECK(m.unsupported.empty());
        }
    }
}

TEST_CASE("standard_surface's subsurface scale and radius collapse into OpenPBR's pair")
{
    const oka::mtlx::MaterialXDocumentData doc = oka::mtlx::loadMaterialXDocument(chessDocument());
    const oka::mtlx::MaterialXMaterial* king = find(doc.materials, "M_King_B");
    REQUIRE(king != nullptr);

    // The document says subsurface_scale = 0.003 and drives subsurface_radius
    // from a map. OpenPBR keeps a scalar length and a normalised tint, so the
    // scale has to end up in the length or the medium is a thousand times too
    // thin -- and the map then modulates the tint, which is the slot above.
    CHECK(king->params.subsurface_radius == doctest::Approx(0.003f));
}
