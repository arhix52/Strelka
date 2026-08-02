#include <strelka/sceneloader/iesloader.h>

#include <log.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <sstream>
#include <vector>

namespace oka
{
namespace
{

std::vector<std::string> tokenizeIesFile(const std::string& path)
{
    std::ifstream in(path);
    std::vector<std::string> tokens;
    if (!in)
    {
        return tokens;
    }
    std::string line;
    while (std::getline(in, line))
    {
        // Strip comments and tidy keywords.
        if (!line.empty() && line.back() == '\r')
            line.pop_back();
        // Label lines in the header look like "[TEST] foo" — skip until TILT.
        if (!tokens.empty() || line.rfind("TILT", 0) == 0 || line.rfind("IESNA", 0) == 0 ||
            (!line.empty() && (std::isdigit(static_cast<unsigned char>(line[0])) || line[0] == '-' || line[0] == '+' ||
                               line[0] == '.')))
        {
            std::istringstream ls(line);
            std::string t;
            while (ls >> t)
            {
                tokens.push_back(t);
            }
        }
    }
    return tokens;
}

} // namespace

bool loadIesProfile(const std::string& path, Scene::IesProfile& out)
{
    const std::vector<std::string> tokens = tokenizeIesFile(path);
    if (tokens.empty())
    {
        STRELKA_ERROR("Failed to open or parse IES file: {}", path);
        return false;
    }

    size_t i = 0;
    // Skip IESNA keyword / version tokens until TILT=.
    while (i < tokens.size() && tokens[i].rfind("TILT", 0) != 0)
    {
        ++i;
    }
    if (i >= tokens.size())
    {
        STRELKA_ERROR("IES file missing TILT=: {}", path);
        return false;
    }

    const std::string& tilt = tokens[i++];
    if (tilt != "TILT=NONE" && tilt != "TILT=None" && tilt != "TILT=none")
    {
        // Embedded tilt: lamp-to-luminaire, n, n angles, n multipliers.
        if (i + 1 >= tokens.size())
            return false;
        ++i; // lamp-to-luminaire
        const int nTilt = std::atoi(tokens[i++].c_str());
        i += (size_t)std::max(nTilt, 0) * 2;
    }

    auto nextFloat = [&](float& v) -> bool {
        if (i >= tokens.size())
            return false;
        v = std::strtof(tokens[i++].c_str(), nullptr);
        return true;
    };
    auto nextInt = [&](int& v) -> bool {
        if (i >= tokens.size())
            return false;
        v = std::atoi(tokens[i++].c_str());
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
    out = std::move(profile);
    STRELKA_INFO("Loaded IES profile {} ({}×{}, max {:.1f} cd)", path, nVertical, nHorizontal, maxC);
    return true;
}

float sampleIesCandela(const Scene::IesProfile& profile, const glm::float3& localDir)
{
    if (profile.candela.empty() || profile.verticalAngles.empty() || profile.horizontalAngles.empty())
    {
        return 0.0f;
    }

    const glm::float3 d = glm::normalize(localDir);
    const float vertDeg = std::acos(std::clamp(-d.z, -1.0f, 1.0f)) * (180.0f / float(M_PI));
    float horizDeg = std::atan2(d.x, -d.y) * (180.0f / float(M_PI));
    if (horizDeg < 0.0f)
        horizDeg += 360.0f;

    const auto& vAng = profile.verticalAngles;
    const auto& hAng = profile.horizontalAngles;
    const int nV = (int)vAng.size();
    const int nH = (int)hAng.size();

    auto lowerIndex = [](const std::vector<float>& a, float x) -> int {
        const auto it = std::lower_bound(a.begin(), a.end(), x);
        if (it == a.begin())
            return 0;
        if (it == a.end())
            return std::max(0, (int)a.size() - 2);
        return (int)(it - a.begin()) - 1;
    };

    const int iv = std::max(0, std::min(nV - 2, lowerIndex(vAng, vertDeg)));
    int ih = 0;
    float th = 0.0f;
    if (nH > 1)
    {
        float h = horizDeg;
        const float hMax = hAng.back();
        if (hMax <= 90.0f + 1e-3f)
            h = std::fmod(h, 90.0f);
        else if (hMax <= 180.0f + 1e-3f)
        {
            if (h > 180.0f)
                h = 360.0f - h;
        }
        else
            h = std::fmod(h, 360.0f);
        ih = std::max(0, std::min(nH - 2, lowerIndex(hAng, h)));
        const float h0 = hAng[(size_t)ih];
        const float h1 = hAng[(size_t)ih + 1];
        th = (h1 > h0) ? (h - h0) / (h1 - h0) : 0.0f;
    }

    const float v0 = vAng[(size_t)iv];
    const float v1 = vAng[(size_t)iv + 1];
    const float tv = (v1 > v0) ? (vertDeg - v0) / (v1 - v0) : 0.0f;

    auto at = [&](int v, int h) { return profile.candela[(size_t)v + (size_t)h * (size_t)nV]; };
    const int ih1 = (nH == 1) ? 0 : ih + 1;
    const float c0 = at(iv, ih) * (1.0f - tv) + at(iv + 1, ih) * tv;
    const float c1 = at(iv, ih1) * (1.0f - tv) + at(iv + 1, ih1) * tv;
    return c0 * (1.0f - th) + c1 * th;
}

} // namespace oka
