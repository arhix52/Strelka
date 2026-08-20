#include <strelka/sceneloader/iesloader.h>

#include <log.h>

// The fold and the clamp the GPU backends apply, so a profile reads the same
// whichever side of the upload it is evaluated on.
#include <ies_math.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <sstream>
#include <vector>
#include <numbers>

namespace oka
{
namespace
{

std::string trimmed(const std::string& line)
{
    size_t b = 0;
    size_t e = line.size();
    while (b < e && std::isspace(static_cast<unsigned char>(line[b])))
    {
        ++b;
    }
    while (e > b && std::isspace(static_cast<unsigned char>(line[e - 1])))
    {
        --e;
    }
    return line.substr(b, e - b);
}

std::string upperCased(std::string v)
{
    for (char& c : v)
    {
        c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    }
    return v;
}

/// One LM-63 file, split at the TILT line.
///
/// The header above TILT is free text -- a version line that may or may not be
/// there, and any number of [KEYWORD] lines -- and everything below it is
/// numbers. Splitting on TILT is the only reliable way to tell the two apart,
/// and it is what the format is structured around.
///
/// The previous reader instead guessed line by line, taking a line if it began
/// with a digit, a sign, a dot, "IESNA" or "TILT" -- and then, once it had taken
/// anything at all, taking every line after it. Three things fell out of that:
/// a file whose keyword text happened to begin with those four letters
/// ("[TESTLAB] TILTON Photometrics") had that word mistaken for the TILT
/// specification; a UTF-8 BOM in front of an otherwise valid TILT line made the
/// line invisible; and TILT written in lower case, or spaced out as "TILT =
/// NONE", was not found at all.
struct IesFileBody
{
    bool found = false;
    std::string tiltSpec; ///< upper-cased text after the '=', e.g. NONE or INCLUDE
    std::vector<std::string> tokens; ///< every whitespace-separated token below the TILT line
};

IesFileBody splitIesFile(const std::string& path)
{
    IesFileBody body;
    std::ifstream in(path);
    if (!in)
    {
        return body;
    }

    std::string line;
    bool first = true;
    while (std::getline(in, line))
    {
        if (!line.empty() && line.back() == '\r')
        {
            line.pop_back();
        }
        if (first)
        {
            // A byte-order mark is invisible to a human editor and turns the
            // first line into something that starts with three high bytes.
            if (line.size() >= 3 && (unsigned char)line[0] == 0xEF && (unsigned char)line[1] == 0xBB &&
                (unsigned char)line[2] == 0xBF)
            {
                line.erase(0, 3);
            }
            first = false;
        }

        if (!body.found)
        {
            const std::string upper = upperCased(trimmed(line));
            if (!upper.starts_with("TILT"))
            {
                continue; // still in the free-text header
            }
            body.found = true;
            const size_t eq = upper.find('=');
            body.tiltSpec = (eq == std::string::npos) ? std::string() : trimmed(upper.substr(eq + 1));
            continue;
        }

        std::istringstream ls(line);
        std::string token;
        while (ls >> token)
        {
            body.tokens.push_back(token);
        }
    }
    return body;
}

/// Unfold a symmetric azimuth table to the full turn, and close it at 360.
///
/// LM-63 encodes the symmetry in the last horizontal angle: 0 means the
/// luminaire is rotationally symmetric and only one column is stored, 90 that
/// it is symmetric about both vertical planes, 180 about one, and 360 that the
/// whole turn is tabulated.
///
/// Doing this once here, rather than folding the lookup angle back at every
/// sample, is what lets the cubic interpolation in ies_math.h have real
/// neighbours at the seams. It is also what Cycles does
/// (IESFile::process_type_c), so the two agree on the columns as well as on the
/// curve drawn through them.
void unfoldAzimuthTable(std::vector<float>& hAngles, std::vector<float>& candela, int nV)
{
    using Diff = std::vector<float>::difference_type;
    const auto column = [&](int h) {
        return std::vector<float>(candela.begin() + Diff(h) * Diff(nV),
                                  candela.begin() + Diff(h + 1) * Diff(nV));
    };
    const auto rebuild = [&](const std::vector<std::vector<float>>& columns) {
        candela.clear();
        candela.reserve(columns.size() * (size_t)nV);
        for (const std::vector<float>& c : columns)
        {
            candela.insert(candela.end(), c.begin(), c.end());
        }
    };

    std::vector<std::vector<float>> columns;
    columns.reserve(hAngles.size());
    for (size_t h = 0; h < hAngles.size(); ++h)
    {
        columns.push_back(column(static_cast<int>(h)));
    }

    // Rotationally symmetric: one column covers everything.
    if (columns.size() == 1)
    {
        hAngles = { 0.0f, 360.0f };
        columns.push_back(columns[0]);
        rebuild(columns);
        return;
    }

    const auto closeTo = [](float a, float b) { return std::fabs(a - b) < 1e-3f; };

    // One quadrant: mirror it about the 90 degree plane to reach 180.
    if (closeTo(hAngles.back(), 90.0f))
    {
        const int n = (int)hAngles.size();
        for (int i = n - 2; i >= 0; --i)
        {
            hAngles.push_back(180.0f - hAngles[(size_t)i]);
            columns.push_back(columns[(size_t)i]);
        }
    }

    // Half: mirror about the 180 degree plane to reach the full turn.
    if (closeTo(hAngles.back(), 180.0f))
    {
        const int n = (int)hAngles.size();
        for (int i = n - 2; i >= 0; --i)
        {
            hAngles.push_back(360.0f - hAngles[(size_t)i]);
            columns.push_back(columns[(size_t)i]);
        }
    }

    // Some files omit the 360 entry because it repeats 0. Close the loop when
    // the spacing says one is missing, so the seam interpolates instead of
    // clamping.
    if (closeTo(hAngles.front(), 0.0f) && !closeTo(hAngles.back(), 360.0f))
    {
        const size_t n = hAngles.size();
        const float lastStep = hAngles[n - 1] - hAngles[n - 2];
        const float firstStep = hAngles[1] - hAngles[0];
        const float gap = 360.0f - hAngles[n - 1];
        if (closeTo(lastStep, gap) || closeTo(firstStep, gap))
        {
            hAngles.push_back(360.0f);
            columns.push_back(columns[0]);
        }
    }

    rebuild(columns);
}

} // namespace

bool loadIesProfile(const std::string& path, Scene::IesProfile& out)
{
    const IesFileBody body = splitIesFile(path);
    if (!body.found)
    {
        STRELKA_ERROR("IES file missing a TILT line (or unreadable): {}", path);
        return false;
    }

    const std::vector<std::string>& tokens = body.tokens;
    size_t i = 0;

    // Three legal specifications: NONE, INCLUDE followed by an embedded block,
    // or the name of a .TLT file holding that block. The tilt data describes how
    // the luminaire's output changes as it is rotated, which this renderer does
    // not model, so all three end the same way -- but INCLUDE has to have its
    // block stepped over, and a file name must NOT be treated as one. Reading a
    // named file's photometric header as a tilt block consumed the lamp count as
    // a pair count and threw the rest of the parse away.
    if (body.tiltSpec == "INCLUDE")
    {
        if (i + 1 >= tokens.size())
        {
            STRELKA_ERROR("IES TILT=INCLUDE block truncated: {}", path);
            return false;
        }
        ++i; // lamp-to-luminaire geometry
        char* end = nullptr;
        const long nTilt = std::strtol(tokens[i].c_str(), &end, 10);
        if (end == tokens[i].c_str() || nTilt < 0)
        {
            STRELKA_ERROR("IES tilt count '{}' is not a count in {}", tokens[i], path);
            return false;
        }
        ++i;
        i += (size_t)nTilt * 2; // angles, then multiplying factors
    }

    auto nextFloat = [&](float& v) -> bool {
        if (i >= tokens.size())
            return false;
        v = std::strtof(tokens[i++].c_str(), nullptr);
        return true;
    };
    // A token that is not a number reads back as 0 through atoi, and a 0 here is
    // a grid dimension or a lamp count the rest of the parse then trusts. Say so
    // instead: a malformed header is a file this loader cannot handle.
    auto nextInt = [&](int& v) -> bool {
        if (i >= tokens.size())
            return false;
        const std::string& token = tokens[i++];
        char* end = nullptr;
        const long parsed = std::strtol(token.c_str(), &end, 10);
        if (end == token.c_str() || parsed < std::numeric_limits<int>::min() || parsed > std::numeric_limits<int>::max())
        {
            STRELKA_ERROR("IES '{}' is not an integer in {}", token, path);
            return false;
        }
        v = static_cast<int>(parsed);
        return true;
    };

    int lamps = 0, nVertical = 0, nHorizontal = 0, photometricType = 1, unitsType = 1;
    float lumensPerLamp = 0, candelaMultiplier = 1, width = 0, length = 0, height = 0;
    float ballast = 1, unused = 1, inputWatts = 0;
    if (!nextInt(lamps) || !nextFloat(lumensPerLamp) || !nextFloat(candelaMultiplier) || !nextInt(nVertical) ||
        !nextInt(nHorizontal) || !nextInt(photometricType) || !nextInt(unitsType) || !nextFloat(width) ||
        !nextFloat(length) || !nextFloat(height) || !nextFloat(ballast) || !nextFloat(unused) || !nextFloat(inputWatts))
    {
        STRELKA_ERROR("IES photometric header unreadable: {}", path);
        return false;
    }
    if (nVertical <= 0 || nHorizontal <= 0 || nVertical > 4096 || nHorizontal > 4096)
    {
        STRELKA_ERROR("IES grid size out of range ({}×{}) in {}", nVertical, nHorizontal, path);
        return false;
    }
    // Type C is the architectural convention and the one sampleIesCandela()
    // implements: the vertical angle is measured from the photometric axis and
    // the azimuth turns around it. Types A and B (2 and 3) tabulate a different
    // pair of angles entirely -- they are for headlamps and floodlights -- and
    // reading one as type C silently rotates the distribution. Loading it anyway
    // is better than refusing the file, but it must not be silent.
    if (photometricType != 1)
    {
        STRELKA_WARNING(
            "IES file {} is photometric type {} (not type C); its angles will be read as "
            "type C, which is only correct for type C files",
            path, photometricType);
    }

    Scene::IesProfile profile;
    profile.path = path;
    profile.verticalAngles.resize((size_t)nVertical);
    profile.horizontalAngles.resize((size_t)nHorizontal);
    profile.candela.resize((size_t)nVertical * (size_t)nHorizontal);

    for (int v = 0; v < nVertical; ++v)
    {
        if (!nextFloat(profile.verticalAngles[(size_t)v]))
        {
            STRELKA_ERROR("IES vertical angles truncated: {}", path);
            return false;
        }
    }
    for (int h = 0; h < nHorizontal; ++h)
    {
        if (!nextFloat(profile.horizontalAngles[(size_t)h]))
        {
            STRELKA_ERROR("IES horizontal angles truncated: {}", path);
            return false;
        }
    }

    // Both tables must ascend: the interval search below is a binary search, and
    // on a descending table it returns an interval that does not contain the
    // angle -- so the profile would evaluate to a plausible-looking wrong number
    // rather than fail.
    for (int v = 1; v < nVertical; ++v)
    {
        if (profile.verticalAngles[(size_t)v] < profile.verticalAngles[(size_t)v - 1])
        {
            STRELKA_ERROR("IES vertical angles are not ascending in {}", path);
            return false;
        }
    }
    for (int h = 1; h < nHorizontal; ++h)
    {
        if (profile.horizontalAngles[(size_t)h] < profile.horizontalAngles[(size_t)h - 1])
        {
            STRELKA_ERROR("IES horizontal angles are not ascending in {}", path);
            return false;
        }
    }

    float maxC = 0.0f;
    for (int h = 0; h < nHorizontal; ++h)
    {
        for (int v = 0; v < nVertical; ++v)
        {
            float c = 0.0f;
            if (!nextFloat(c))
            {
                STRELKA_ERROR("IES candela values truncated: {}", path);
                return false;
            }
            c *= candelaMultiplier * ballast;
            profile.candela[(size_t)v + (size_t)h * (size_t)nVertical] = c;
            maxC = std::max(maxC, c);
        }
    }
    profile.maxCandela = maxC;
    unfoldAzimuthTable(profile.horizontalAngles, profile.candela, nVertical);
    out = std::move(profile);
    STRELKA_INFO("Loaded IES profile {} ({}×{}, max {:.1f} cd)", path, nVertical, nHorizontal, maxC);
    return true;
}

void unfoldIesAzimuth(Scene::IesProfile& profile)
{
    if (profile.verticalAngles.empty() || profile.horizontalAngles.empty())
    {
        return;
    }
    unfoldAzimuthTable(profile.horizontalAngles, profile.candela, (int)profile.verticalAngles.size());
}

float sampleIesCandela(const Scene::IesProfile& profile, const glm::float3& localDir)
{
    if (profile.candela.empty() || profile.verticalAngles.empty() || profile.horizontalAngles.empty())
    {
        return 0.0f;
    }

    const glm::float3 d = glm::normalize(localDir);
    const float vertDeg = std::acos(std::clamp(-d.z, -1.0f, 1.0f)) * (180.0f / std::numbers::pi_v<float>);
    const float horizDeg = std::atan2(d.x, -d.y) * (180.0f / std::numbers::pi_v<float>);

    // The same evaluation the GPU runs, from the same header -- see ies_math.h.
    return iesEvaluate(profile.verticalAngles.data(), (int)profile.verticalAngles.size(), profile.horizontalAngles.data(),
                       (int)profile.horizontalAngles.size(), profile.candela.data(), vertDeg, horizDeg);
}

} // namespace oka
