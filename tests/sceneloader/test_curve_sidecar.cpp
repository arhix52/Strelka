#include <doctest/doctest.h>

#include <strelka/scene/scene.h>
#include <strelka/sceneloader/curve_sidecar.h>

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

// Curves reach the renderer through a binary sidecar, and nothing between the
// exporter and the acceleration structure validates them: a set whose strand
// counts disagree with its point array does not draw wrong, it builds a corrupt
// BLAS. So the reader is the last place a bad file can be caught, and these
// cases are what it is required to catch or carry.
//
// The radii are the reason this file exists. The format carries *radii*, while
// Blender's particle properties are diameters -- an exporter that skipped the
// conversion rendered every groom at twice its reference thickness, which
// presented as a warm shading error rather than a geometric one (see
// docs/open-defects.md, Closed). The writer end of that contract is pinned in
// tools/iso_bathroom/test_curve_sidecar.py; this end pins that the reader hands
// on whatever it was given, unscaled, so there is exactly one place the
// convention can be wrong.

using namespace oka;
namespace fs = std::filesystem;

namespace oka
{
// Defined in gltfloader.cpp and not declared in any header, because the glTF
// loader is its only caller. Redeclared here rather than exported: the naming
// convention it implements is shared with two Python exporters that hardcode the
// same suffix, and if the two ever disagree the result is a scene that loads
// clean with no hair in it.
bool loadCurvesFromSidecar(const std::string& modelPath, oka::Scene& scene);
} // namespace oka

namespace
{

struct Strand
{
    std::vector<glm::float3> points;
    std::vector<float> radii;
};

struct SetSpec
{
    std::string material = "hair0";
    uint32_t basis = 0; // 0 linear, 1 cubic B-spline
    std::vector<Strand> strands;
    glm::mat4 transform = glm::mat4(1.0f);
};

/// A strand of `n` control points climbing +Y, with a taper if two radii differ.
Strand makeStrand(uint32_t n, float rootRadius = 0.002f, float tipRadius = 0.002f, float x = 0.0f)
{
    Strand s;
    for (uint32_t i = 0; i < n; ++i)
    {
        const float t = (n > 1) ? float(i) / float(n - 1) : 0.0f;
        s.points.emplace_back(x, t, 0.0f);
        s.radii.push_back(rootRadius + (tipRadius - rootRadius) * t);
    }
    return s;
}

/// The documented layout, assembled longhand rather than by calling the Python
/// writer: if the two drift apart, that is the thing worth failing on.
struct Blob
{
    std::vector<char> bytes;

    void raw(const void* p, size_t n)
    {
        const char* c = static_cast<const char*>(p);
        bytes.insert(bytes.end(), c, c + n);
    }
    void u32(uint32_t v)
    {
        raw(&v, sizeof(v));
    }
    void f32(float v)
    {
        raw(&v, sizeof(v));
    }
};

Blob buildSidecar(const std::vector<SetSpec>& sets, const char* magic = nullptr)
{
    Blob b;
    if (magic != nullptr)
    {
        b.raw(magic, 8);
    }
    else
    {
        b.raw(curvesidecar::kMagic, 8);
    }
    b.u32(static_cast<uint32_t>(sets.size()));
    for (const SetSpec& s : sets)
    {
        uint32_t pointCount = 0;
        for (const Strand& st : s.strands)
        {
            pointCount += static_cast<uint32_t>(st.points.size());
        }

        b.u32(static_cast<uint32_t>(s.material.size()));
        b.raw(s.material.data(), s.material.size());
        b.u32(s.basis);
        b.u32(static_cast<uint32_t>(s.strands.size()));
        b.u32(pointCount);
        b.raw(&s.transform[0][0], sizeof(float) * 16);
        for (const Strand& st : s.strands)
        {
            b.u32(static_cast<uint32_t>(st.points.size()));
        }
        for (const Strand& st : s.strands)
        {
            for (const glm::float3& p : st.points)
            {
                b.f32(p.x);
                b.f32(p.y);
                b.f32(p.z);
            }
        }
        for (const Strand& st : s.strands)
        {
            for (const float r : st.radii)
            {
                b.f32(r);
            }
        }
    }
    return b;
}

std::string writeBlob(const Blob& b, const char* stem, size_t truncateTo = 0)
{
    const fs::path p = fs::temp_directory_path() / stem;
    std::ofstream f(p, std::ios::binary | std::ios::trunc);
    const size_t n = (truncateTo == 0 || truncateTo > b.bytes.size()) ? b.bytes.size() : truncateTo;
    f.write(b.bytes.data(), static_cast<std::streamsize>(n));
    f.close();
    return p.string();
}

uint32_t addNamedMaterial(Scene& scene, const char* name)
{
    Scene::MaterialDescription d;
    d.name = name;
    return scene.addMaterial(d);
}

} // namespace

TEST_CASE("Curve sidecar carries points and radii through unscaled")
{
    Scene scene;
    addNamedMaterial(scene, "hair0");

    SetSpec set;
    set.material = "hair0";
    // The 28_hair groom's own numbers, after the diameter -> radius conversion.
    set.strands = { makeStrand(9, 0.002f, 0.00075f, 0.0f), makeStrand(9, 0.002f, 0.00075f, 1.0f) };
    const std::string path = writeBlob(buildSidecar({ set }), "strelka_curves_roundtrip.bin");

    REQUIRE(curvesidecar::loadCurvesFile(path, scene));
    REQUIRE(scene.getCurves().size() == 1);

    const Curve& c = scene.getCurves()[0];
    CHECK(c.mPointsCount == 18);
    CHECK(c.mVertexCountsCount == 2);
    CHECK(c.mWidthsCount == 18);

    const std::vector<float>& widths = scene.getCurvesWidths();
    REQUIRE(widths.size() >= 18);
    // Root and tip verbatim: the reader is not allowed to reinterpret the unit.
    CHECK(widths[c.mWidthsStart + 0] == doctest::Approx(0.002f));
    CHECK(widths[c.mWidthsStart + 8] == doctest::Approx(0.00075f));
    // And the taper in between is monotonic rather than flattened to one value,
    // which is what a radius buffer bound with the wrong stride would look like.
    for (uint32_t i = 1; i < 9; ++i)
    {
        CHECK(widths[c.mWidthsStart + i] < widths[c.mWidthsStart + i - 1]);
    }

    const std::vector<glm::float3>& points = scene.getCurvesPoint();
    REQUIRE(points.size() >= 18);
    CHECK(points[c.mPointsStart + 0].x == doctest::Approx(0.0f));
    CHECK(points[c.mPointsStart + 8].y == doctest::Approx(1.0f));
    CHECK(points[c.mPointsStart + 9].x == doctest::Approx(1.0f));
}

TEST_CASE("Curve sidecar basis selects the curve type")
{
    SUBCASE("0 is linear")
    {
        Scene scene;
        addNamedMaterial(scene, "hair0");
        SetSpec set;
        set.basis = 0;
        set.strands = { makeStrand(4) };
        const std::string path = writeBlob(buildSidecar({ set }), "strelka_curves_linear.bin");
        REQUIRE(curvesidecar::loadCurvesFile(path, scene));
        CHECK(scene.getCurves()[0].mType == Curve::Type::eLinear);
    }

    SUBCASE("1 is cubic B-spline")
    {
        Scene scene;
        addNamedMaterial(scene, "hair0");
        SetSpec set;
        set.basis = 1;
        set.strands = { makeStrand(4) };
        const std::string path = writeBlob(buildSidecar({ set }), "strelka_curves_cubic.bin");
        REQUIRE(curvesidecar::loadCurvesFile(path, scene));
        CHECK(scene.getCurves()[0].mType == Curve::Type::eCubic);
    }
}

// mSegmentsPerStrand is what lets a shader recover how far along a strand a hit
// landed from the segment index alone, so it is a root-to-tip gradient for no
// memory -- and a wrong value silently shades the gradient backwards or flat.
TEST_CASE("Curve sidecar derives segments per strand only when the set is uniform")
{
    SUBCASE("uniform linear: one segment per gap")
    {
        Scene scene;
        addNamedMaterial(scene, "hair0");
        SetSpec set;
        set.basis = 0;
        set.strands = { makeStrand(9), makeStrand(9), makeStrand(9) };
        const std::string path = writeBlob(buildSidecar({ set }), "strelka_curves_seg_linear.bin");
        REQUIRE(curvesidecar::loadCurvesFile(path, scene));
        CHECK(scene.getCurves()[0].mSegmentsPerStrand == 8);
    }

    SUBCASE("uniform cubic: a segment spans four control points")
    {
        Scene scene;
        addNamedMaterial(scene, "hair0");
        SetSpec set;
        set.basis = 1;
        set.strands = { makeStrand(8), makeStrand(8) };
        const std::string path = writeBlob(buildSidecar({ set }), "strelka_curves_seg_cubic.bin");
        REQUIRE(curvesidecar::loadCurvesFile(path, scene));
        CHECK(scene.getCurves()[0].mSegmentsPerStrand == 5);
    }

    SUBCASE("mixed strand lengths give up the gradient rather than guess")
    {
        Scene scene;
        addNamedMaterial(scene, "hair0");
        SetSpec set;
        set.strands = { makeStrand(9), makeStrand(5) };
        const std::string path = writeBlob(buildSidecar({ set }), "strelka_curves_seg_mixed.bin");
        REQUIRE(curvesidecar::loadCurvesFile(path, scene));
        CHECK(scene.getCurves()[0].mSegmentsPerStrand == 0);
        // The set still loads: a mixed groom is renderable, just ungraded.
        CHECK(scene.getCurves()[0].mVertexCountsCount == 2);
    }

    SUBCASE("a strand too short for one segment is not a segment")
    {
        Scene scene;
        addNamedMaterial(scene, "hair0");
        SetSpec set;
        set.basis = 0;
        set.strands = { makeStrand(1), makeStrand(1) };
        const std::string path = writeBlob(buildSidecar({ set }), "strelka_curves_seg_short.bin");
        REQUIRE(curvesidecar::loadCurvesFile(path, scene));
        CHECK(scene.getCurves()[0].mSegmentsPerStrand == 0);
    }

    SUBCASE("cubic needs four points, so three is still zero")
    {
        Scene scene;
        addNamedMaterial(scene, "hair0");
        SetSpec set;
        set.basis = 1;
        set.strands = { makeStrand(3) };
        const std::string path = writeBlob(buildSidecar({ set }), "strelka_curves_seg_cubic3.bin");
        REQUIRE(curvesidecar::loadCurvesFile(path, scene));
        CHECK(scene.getCurves()[0].mSegmentsPerStrand == 0);
    }
}

TEST_CASE("Curve sidecar binds strands to a material by name")
{
    SUBCASE("a name that exists")
    {
        Scene scene;
        addNamedMaterial(scene, "stage");
        const uint32_t hairId = addNamedMaterial(scene, "hair0");
        SetSpec set;
        set.material = "hair0";
        set.strands = { makeStrand(4) };
        const std::string path = writeBlob(buildSidecar({ set }), "strelka_curves_mat_ok.bin");

        REQUIRE(curvesidecar::loadCurvesFile(path, scene));
        REQUIRE(scene.mInstances.size() == 1);
        CHECK(scene.mInstances[0].mMaterialId == hairId);
        CHECK(hairId != 0); // otherwise the fallback below would pass by accident
    }

    SUBCASE("a name that does not falls back to material 0 and still loads")
    {
        Scene scene;
        addNamedMaterial(scene, "stage");
        SetSpec set;
        set.material = "no_such_material";
        set.strands = { makeStrand(4) };
        const std::string path = writeBlob(buildSidecar({ set }), "strelka_curves_mat_missing.bin");

        REQUIRE(curvesidecar::loadCurvesFile(path, scene));
        REQUIRE(scene.mInstances.size() == 1);
        CHECK(scene.mInstances[0].mMaterialId == 0);
    }
}

TEST_CASE("Curve sidecar creates a curve instance carrying its transform")
{
    Scene scene;
    addNamedMaterial(scene, "hair0");

    SetSpec set;
    set.strands = { makeStrand(4) };
    set.transform = glm::mat4(1.0f);
    set.transform[3][0] = 3.0f; // column-major: translation lives in column 3
    set.transform[3][1] = -2.0f;
    set.transform[3][2] = 0.5f;
    const std::string path = writeBlob(buildSidecar({ set }), "strelka_curves_xform.bin");

    REQUIRE(curvesidecar::loadCurvesFile(path, scene));
    REQUIRE(scene.mInstances.size() == 1);

    const Instance& inst = scene.mInstances[0];
    CHECK(inst.type == Instance::Type::eCurve);
    CHECK(inst.mCurveId == 0);
    CHECK(inst.transform[3][0] == doctest::Approx(3.0f));
    CHECK(inst.transform[3][1] == doctest::Approx(-2.0f));
    CHECK(inst.transform[3][2] == doctest::Approx(0.5f));
}

TEST_CASE("Curve sidecar keeps several sets separate")
{
    Scene scene;
    const uint32_t aId = addNamedMaterial(scene, "hair_a");
    const uint32_t bId = addNamedMaterial(scene, "hair_b");

    SetSpec a;
    a.material = "hair_a";
    a.basis = 0;
    a.strands = { makeStrand(9), makeStrand(9) };

    SetSpec b;
    b.material = "hair_b";
    b.basis = 1;
    b.strands = { makeStrand(8) };

    const std::string path = writeBlob(buildSidecar({ a, b }), "strelka_curves_multi.bin");
    REQUIRE(curvesidecar::loadCurvesFile(path, scene));

    REQUIRE(scene.getCurves().size() == 2);
    REQUIRE(scene.mInstances.size() == 2);

    CHECK(scene.getCurves()[0].mType == Curve::Type::eLinear);
    CHECK(scene.getCurves()[0].mSegmentsPerStrand == 8);
    CHECK(scene.getCurves()[1].mType == Curve::Type::eCubic);
    CHECK(scene.getCurves()[1].mSegmentsPerStrand == 5);
    CHECK(scene.mInstances[0].mMaterialId == aId);
    CHECK(scene.mInstances[1].mMaterialId == bId);

    // The second set's payload must start where the first one's ended, or the
    // acceleration structure indexes into the wrong groom.
    const Curve& second = scene.getCurves()[1];
    CHECK(second.mPointsStart == 18);
    CHECK(second.mWidthsStart == 18);
    CHECK(second.mVertexCountsStart == 2);
}

TEST_CASE("Curve sidecar rejects a file it cannot trust")
{
    SetSpec set;
    set.strands = { makeStrand(9), makeStrand(9) };
    const Blob good = buildSidecar({ set });

    SUBCASE("a missing file")
    {
        Scene scene;
        addNamedMaterial(scene, "hair0");
        const fs::path missing = fs::temp_directory_path() / "strelka_curves_does_not_exist.bin";
        fs::remove(missing);
        CHECK_FALSE(curvesidecar::loadCurvesFile(missing.string(), scene));
        CHECK(scene.getCurves().empty());
    }

    SUBCASE("the wrong magic")
    {
        Scene scene;
        addNamedMaterial(scene, "hair0");
        const std::string path =
            writeBlob(buildSidecar({ set }, "NOTACRV0"), "strelka_curves_bad_magic.bin");
        CHECK_FALSE(curvesidecar::loadCurvesFile(path, scene));
        CHECK(scene.getCurves().empty());
    }

    SUBCASE("truncated before the set count")
    {
        Scene scene;
        addNamedMaterial(scene, "hair0");
        const std::string path = writeBlob(good, "strelka_curves_trunc_count.bin", 10);
        CHECK_FALSE(curvesidecar::loadCurvesFile(path, scene));
    }

    SUBCASE("truncated inside the set header")
    {
        Scene scene;
        addNamedMaterial(scene, "hair0");
        const std::string path = writeBlob(good, "strelka_curves_trunc_header.bin", 24);
        CHECK_FALSE(curvesidecar::loadCurvesFile(path, scene));
    }

    SUBCASE("truncated inside the payload")
    {
        Scene scene;
        addNamedMaterial(scene, "hair0");
        const std::string path =
            writeBlob(good, "strelka_curves_trunc_payload.bin", good.bytes.size() - 16);
        CHECK_FALSE(curvesidecar::loadCurvesFile(path, scene));
    }

    SUBCASE("strand counts that do not add up to the point array")
    {
        // The case that would otherwise reach the driver: every field is present
        // and self-consistent except the arithmetic between them.
        Scene scene;
        addNamedMaterial(scene, "hair0");
        Blob b;
        b.raw(curvesidecar::kMagic, 8);
        b.u32(1);
        const std::string name = "hair0";
        b.u32(static_cast<uint32_t>(name.size()));
        b.raw(name.data(), name.size());
        b.u32(0); // linear
        b.u32(2); // two strands
        b.u32(6); // six points claimed
        const glm::mat4 identity(1.0f);
        b.raw(&identity[0][0], sizeof(float) * 16);
        b.u32(4); // but the counts say 4 + 4
        b.u32(4);
        for (int i = 0; i < 6 * 3; ++i)
        {
            b.f32(0.0f);
        }
        for (int i = 0; i < 6; ++i)
        {
            b.f32(0.002f);
        }
        const std::string path = writeBlob(b, "strelka_curves_count_mismatch.bin");
        CHECK_FALSE(curvesidecar::loadCurvesFile(path, scene));
    }

    SUBCASE("a file that declares no sets at all")
    {
        Scene scene;
        addNamedMaterial(scene, "hair0");
        const std::string path = writeBlob(buildSidecar({}), "strelka_curves_no_sets.bin");
        CHECK_FALSE(curvesidecar::loadCurvesFile(path, scene));
        CHECK(scene.getCurves().empty());
    }
}

TEST_CASE("Curves are found beside the model, by the model's name")
{
    const fs::path dir = fs::temp_directory_path() / "strelka_curve_lookup";
    fs::remove_all(dir);
    fs::create_directories(dir);

    SetSpec set;
    set.strands = { makeStrand(9), makeStrand(9) };
    const Blob blob = buildSidecar({ set });

    const auto write = [&](const fs::path& p) {
        std::ofstream f(p, std::ios::binary | std::ios::trunc);
        f.write(blob.bytes.data(), static_cast<std::streamsize>(blob.bytes.size()));
    };

    SUBCASE("<stem>_curves.bin next to a .gltf")
    {
        write(dir / "groom_curves.bin");
        Scene scene;
        addNamedMaterial(scene, "hair0");
        REQUIRE(loadCurvesFromSidecar((dir / "groom.gltf").string(), scene));
        CHECK(scene.getCurves().size() == 1);
    }

    SUBCASE("and next to a .glb, since only the extension is stripped")
    {
        write(dir / "groom_curves.bin");
        Scene scene;
        addNamedMaterial(scene, "hair0");
        REQUIRE(loadCurvesFromSidecar((dir / "groom.glb").string(), scene));
        CHECK(scene.getCurves().size() == 1);
    }

    SUBCASE("a sidecar belonging to another model in the same directory is not taken")
    {
        // The reason the lookup is by name and not a directory scan: two converted
        // scenes sharing a folder would otherwise swap grooms.
        write(dir / "other_curves.bin");
        Scene scene;
        addNamedMaterial(scene, "hair0");
        CHECK_FALSE(loadCurvesFromSidecar((dir / "groom.gltf").string(), scene));
        CHECK(scene.getCurves().empty());
    }

    SUBCASE("no sidecar is the ordinary case, not a failure")
    {
        Scene scene;
        addNamedMaterial(scene, "hair0");
        CHECK_FALSE(loadCurvesFromSidecar((dir / "groom.gltf").string(), scene));
        CHECK(scene.getCurves().empty());
    }

    fs::remove_all(dir);
}

// Scene::createCurve is also reachable from converters that have no radii to
// give, and its answer there is a sentinel rather than zero -- a zero-radius
// curve is invisible, which is a bug that looks like missing geometry.
TEST_CASE("Scene::createCurve marks an absent width array instead of defaulting it")
{
    Scene scene;
    const std::vector<uint32_t> counts = { 4 };
    const std::vector<glm::float3> points = { glm::float3(0.0f), glm::float3(0.0f, 1.0f, 0.0f),
                                             glm::float3(0.0f, 2.0f, 0.0f),
                                             glm::float3(0.0f, 3.0f, 0.0f) };

    const uint32_t withWidths =
        scene.createCurve(Curve::Type::eLinear, counts, points, { 0.1f, 0.2f, 0.3f, 0.4f });
    const uint32_t without = scene.createCurve(Curve::Type::eCubic, counts, points, {});

    CHECK(scene.getCurves()[withWidths].mWidthsCount == 4);
    CHECK(scene.getCurves()[without].mWidthsCount == static_cast<uint32_t>(-1));
    CHECK(scene.getCurves()[without].mWidthsStart == static_cast<uint32_t>(-1));
    // The type is stored rather than assumed: the two backends used to guess it
    // and guess differently.
    CHECK(scene.getCurves()[withWidths].mType == Curve::Type::eLinear);
    CHECK(scene.getCurves()[without].mType == Curve::Type::eCubic);
}
