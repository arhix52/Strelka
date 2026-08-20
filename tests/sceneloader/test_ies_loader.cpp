#include <doctest/doctest.h>

#include <strelka/scene/scene.h>
#include <strelka/sceneloader/iesloader.h>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <string>

// ============================================================================
// test_ies_loader.cpp -- reading an actual LM-63 file.
//
// The suite already had one IES case, and it wrote the smallest file that could
// possibly parse: a version line, TILT=NONE, three angles and three numbers. A
// real photometric file is not that. It carries a block of [KEYWORD] lines whose
// text is arbitrary, a candela multiplier and a ballast factor that both scale
// the table, values wrapped across lines wherever the exporter felt like it, and
// one of three different TILT specifications. Every one of those was a code path
// nothing exercised.
//
// Four of them were broken, and each fails the same way -- loadIesProfile()
// returns false or reads garbage, so the luminaire silently becomes an
// isotropic point light with whatever intensity the sidecar gave it. That is
// what makes them worth a test rather than a look: the failure does not look
// like a failure, it looks like a plain lamp.
// ============================================================================

namespace fs = std::filesystem;

namespace
{

struct TempIes
{
    fs::path path;

    explicit TempIes(const std::string& name, const std::string& body)
        : path(fs::temp_directory_path() / ("strelka_" + name + ".ies"))
    {
        std::ofstream out(path, std::ios::binary);
        out << body;
    }
    ~TempIes()
    {
        std::error_code ec;
        fs::remove(path, ec);
    }

    TempIes(const TempIes&) = delete;
    TempIes& operator=(const TempIes&) = delete;
};

/// The photometric block shared by most cases below: three vertical angles, one
/// horizontal plane, candela falling to zero at the horizon.
const char* const kSimplePhotometry =
    "1 -1 1.0 3 1 1 1 0 0 0\n"
    "1.0 1.0 20\n"
    "0 45 90\n"
    "0\n"
    "800 400 0\n";

/// Direction at a given vertical angle in the light's local frame, on the
/// azimuth the table is tabulated at. -Z is the photometric axis.
glm::float3 atVertical(float deg)
{
    const float th = deg * float(M_PI) / 180.0f;
    return { std::sin(th), 0.0f, -std::cos(th) };
}

glm::float3 atAngles(float verticalDeg, float azimuthDeg)
{
    const float th = verticalDeg * float(M_PI) / 180.0f;
    const float ph = azimuthDeg * float(M_PI) / 180.0f;
    return { std::sin(th) * std::sin(ph), -std::sin(th) * std::cos(ph), -std::cos(th) };
}

} // namespace

TEST_CASE("a realistic LM-63-2002 file parses, multipliers and all")
{
    // Keywords with arbitrary text, a candela multiplier of 0.5, a ballast
    // factor of 1.02, and the candela values wrapped across two lines -- all of
    // which a real file does and the previous test file did not.
    const TempIes f("realistic",
                    "IESNA:LM-63-2002\n"
                    "[TEST] LTL-12345\n"
                    "[TESTLAB] Independent Testing Laboratories\n"
                    "[ISSUEDATE] 12-JUN-2019\n"
                    "[MANUFAC] Acme Lighting\n"
                    "[LUMCAT] ACME-DL-3000K\n"
                    "[LUMINAIRE] 6in LED downlight\n"
                    "[MORE] 3000K 90CRI\n"
                    "TILT=NONE\n"
                    "1 1200 0.5 5 1 1 1 0.5 0.5 0.3\n"
                    "1.02 1.0 12.5\n"
                    "0 22.5 45 67.5 90\n"
                    "0\n"
                    "2000 1800\n"
                    "1200 400 0\n");

    oka::Scene::IesProfile p;
    REQUIRE(oka::loadIesProfile(f.path.string(), p));
    CHECK(p.verticalAngles.size() == 5);
    // One tabulated plane, unfolded to 0 and 360 at load time.
    CHECK(p.horizontalAngles.size() == 2);
    CHECK(p.candela.size() == 10);

    // Both scale factors are applied: 2000 * 0.5 * 1.02.
    CHECK(p.maxCandela == doctest::Approx(1020.0f));
    CHECK(oka::sampleIesCandela(p, atVertical(0.0f)) == doctest::Approx(1020.0f));
    CHECK(oka::sampleIesCandela(p, atVertical(90.0f)) == doctest::Approx(0.0f));

    // And this is a beam, not a bare point: it has a shape between the tabulated
    // angles and it falls off. A profile that came back constant would mean the
    // table was read as a single value and the luminaire had quietly become a
    // point light.
    const float onAxis = oka::sampleIesCandela(p, atVertical(0.0f));
    const float at45 = oka::sampleIesCandela(p, atVertical(45.0f));
    const float at67 = oka::sampleIesCandela(p, atVertical(67.5f));
    CHECK(at45 == doctest::Approx(1200.0f * 0.5f * 1.02f));
    CHECK(at67 == doctest::Approx(400.0f * 0.5f * 1.02f));
    CHECK(at45 < onAxis * 0.7f);
    CHECK(at67 < at45 * 0.5f);
    // Interpolated between rows rather than snapped to one.
    const float at56 = oka::sampleIesCandela(p, atVertical(56.25f));
    CHECK(at56 < at45);
    CHECK(at56 > at67);
}

TEST_CASE("the candela block is laid out with the horizontal plane on the outside")
{
    // LM-63 writes all vertical angles for the first horizontal plane, then all
    // of them for the second, and so on. Transposing it is not a crash -- it is
    // a luminaire whose beam has been rotated into a different plane.
    const TempIes f("planes",
                    "IESNA:LM-63-1995\n"
                    "TILT=NONE\n"
                    "1 -1 1.0 3 4 1 1 0 0 0\n"
                    "1.0 1.0 20\n"
                    "0 45 90\n"
                    "0 30 60 90\n"
                    "900 450 0\n"
                    "800 400 0\n"
                    "700 350 0\n"
                    "600 300 0\n");

    oka::Scene::IesProfile p;
    REQUIRE(oka::loadIesProfile(f.path.string(), p));
    // One quadrant tabulated at 0/30/60/90, mirrored to the full turn: 13
    // columns ending on the 360 duplicate.
    REQUIRE(p.horizontalAngles.size() == 13);
    CHECK(p.horizontalAngles.front() == doctest::Approx(0.0f));
    CHECK(p.horizontalAngles.back() == doctest::Approx(360.0f));

    // 45 degrees down, on each tabulated azimuth: 450, 400, 350, 300.
    CHECK(oka::sampleIesCandela(p, atAngles(45.0f, 0.0f)) == doctest::Approx(450.0f));
    CHECK(oka::sampleIesCandela(p, atAngles(45.0f, 30.0f)) == doctest::Approx(400.0f));
    CHECK(oka::sampleIesCandela(p, atAngles(45.0f, 60.0f)) == doctest::Approx(350.0f));
    CHECK(oka::sampleIesCandela(p, atAngles(45.0f, 90.0f)) == doctest::Approx(300.0f));
}

// ---------------------------------------------------------------------------
// The three TILT specifications
// ---------------------------------------------------------------------------
TEST_CASE("TILT=INCLUDE steps over its embedded block")
{
    const TempIes f("tilt_include",
                    "IESNA:LM-63-1995\n"
                    "TILT=INCLUDE\n"
                    "1\n" // lamp-to-luminaire geometry
                    "3\n" // pairs
                    "0 45 90\n" // angles
                    "1.0 0.9 0.5\n" // multiplying factors
                        + std::string(kSimplePhotometry));

    oka::Scene::IesProfile p;
    REQUIRE(oka::loadIesProfile(f.path.string(), p));
    CHECK(p.verticalAngles.size() == 3);
    CHECK(p.maxCandela == doctest::Approx(800.0f));
}

TEST_CASE("TILT naming an external file is not read as an embedded block")
{
    // The third legal form, and the one that used to break the parse outright:
    // the file name was treated as INCLUDE, so the lamp count was consumed as a
    // tilt-pair count and everything after it was thrown away. The tilt data
    // lives in ACME.TLT, which this renderer does not need -- but the photometry
    // below it does have to be read.
    const TempIes f("tilt_file", "IESNA:LM-63-1995\nTILT=ACME.TLT\n" + std::string(kSimplePhotometry));

    oka::Scene::IesProfile p;
    REQUIRE(oka::loadIesProfile(f.path.string(), p));
    CHECK(p.verticalAngles.size() == 3);
    CHECK(oka::sampleIesCandela(p, atVertical(0.0f)) == doctest::Approx(800.0f));
}

// ---------------------------------------------------------------------------
// The header, as files in the wild actually write it
// ---------------------------------------------------------------------------
TEST_CASE("a keyword whose text begins with TILT is not the TILT line")
{
    // "[TESTLAB] TILTON Photometrics" is a perfectly ordinary keyword line. The
    // reader used to scan for a *token* starting with those four letters and
    // took this one, then failed to match it against TILT=NONE and tried to read
    // an embedded tilt block out of the laboratory's name.
    const TempIes f("tilty",
                    "IESNA:LM-63-2002\n"
                    "[TESTLAB] TILTON Photometrics\n"
                    "TILT=NONE\n" +
                        std::string(kSimplePhotometry));

    oka::Scene::IesProfile p;
    REQUIRE(oka::loadIesProfile(f.path.string(), p));
    CHECK(p.maxCandela == doctest::Approx(800.0f));
}

TEST_CASE("the TILT line survives a BOM, indentation, lower case and spaces")
{
    struct Case
    {
        const char* name;
        std::string header;
    };
    const Case cases[] = {
        // A byte-order mark is invisible in an editor and made the line
        // unrecognisable when there was no version line above it to absorb it.
        { "bom, no version", std::string("\xEF\xBB\xBF") + "TILT=NONE\n" },
        { "bom and version", std::string("\xEF\xBB\xBF") + "IESNA:LM-63-2002\nTILT=NONE\n" },
        { "blank lines first", "\n\n\nIESNA:LM-63-2002\nTILT=NONE\n" },
        { "indented", "IESNA:LM-63-2002\n   TILT=NONE\n" },
        { "lower case", "IESNA:LM-63-2002\ntilt=none\n" },
        { "spaces around =", "IESNA:LM-63-2002\nTILT = NONE\n" },
    };

    int index = 0;
    for (const Case& c : cases)
    {
        CAPTURE(c.name);
        const TempIes f("hdr" + std::to_string(index++), c.header + std::string(kSimplePhotometry));
        oka::Scene::IesProfile p;
        REQUIRE(oka::loadIesProfile(f.path.string(), p));
        CHECK(p.verticalAngles.size() == 3);
        CHECK(p.maxCandela == doctest::Approx(800.0f));
    }
}

TEST_CASE("a file with no version line at all still parses")
{
    // LM-63-1986 has no version identifier; the keyword block simply starts.
    const TempIes f("no_version",
                    "[TEST] LTL-12345\n"
                    "[MANUFAC] Acme Lighting\n"
                    "TILT=NONE\n" +
                        std::string(kSimplePhotometry));

    oka::Scene::IesProfile p;
    REQUIRE(oka::loadIesProfile(f.path.string(), p));
    CHECK(p.maxCandela == doctest::Approx(800.0f));
}

// ---------------------------------------------------------------------------
// Angle ranges
// ---------------------------------------------------------------------------
TEST_CASE("a luminaire tabulated over the upper hemisphere reads right way up")
{
    // Vertical angles 90..180 are legal and describe something aimed upwards.
    const TempIes f("uplight",
                    "IESNA:LM-63-2002\n"
                    "TILT=NONE\n"
                    "1 -1 1.0 3 1 1 1 0 0 0\n"
                    "1.0 1.0 20\n"
                    "90 135 180\n"
                    "0\n"
                    "0 400 800\n");

    oka::Scene::IesProfile p;
    REQUIRE(oka::loadIesProfile(f.path.string(), p));
    CHECK(oka::sampleIesCandela(p, atVertical(180.0f)) == doctest::Approx(800.0f));
    CHECK(oka::sampleIesCandela(p, atVertical(90.0f)) == doctest::Approx(0.0f));
    // Below the tabulated range there is no data, and the clamp holds the edge
    // value rather than extrapolating past it.
    CHECK(oka::sampleIesCandela(p, atVertical(0.0f)) == doctest::Approx(0.0f));
}

TEST_CASE("a descending angle table is refused rather than silently misread")
{
    // The interval search is a binary search. On a table that descends it
    // returns an interval that does not contain the angle, so the profile would
    // evaluate to a plausible-looking wrong number in every direction.
    const TempIes f("descending",
                    "IESNA:LM-63-2002\n"
                    "TILT=NONE\n"
                    "1 -1 1.0 3 1 1 1 0 0 0\n"
                    "1.0 1.0 20\n"
                    "90 45 0\n"
                    "0\n"
                    "0 400 800\n");

    oka::Scene::IesProfile p;
    CHECK_FALSE(oka::loadIesProfile(f.path.string(), p));
}

TEST_CASE("a truncated file is refused rather than read as zeros")
{
    const TempIes f("truncated",
                    "IESNA:LM-63-2002\n"
                    "TILT=NONE\n"
                    "1 -1 1.0 3 1 1 1 0 0 0\n"
                    "1.0 1.0 20\n"
                    "0 45 90\n"
                    "0\n"
                    "800 400\n"); // one candela value short

    oka::Scene::IesProfile p;
    CHECK_FALSE(oka::loadIesProfile(f.path.string(), p));
}

TEST_CASE("a file with no TILT line at all is refused")
{
    const TempIes f("no_tilt", "IESNA:LM-63-2002\n[TEST] 1\n" + std::string(kSimplePhotometry));
    oka::Scene::IesProfile p;
    CHECK_FALSE(oka::loadIesProfile(f.path.string(), p));
}
