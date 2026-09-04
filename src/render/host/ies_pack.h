#pragma once

#include <strelka/scene/light_desc.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

namespace oka::ies_pack
{

struct IesBufferHeader
{
    uint32_t profileCount;
    uint32_t floatOffset;
    uint32_t pad0;
    uint32_t pad1;
};

struct IesProfileHeader
{
    uint32_t nVertical;
    uint32_t nHorizontal;
    uint32_t anglesOffset;
    uint32_t candelaOffset;
    float maxCandela;
    float pad0;
    float pad1;
    float pad2;
};

using oka::kCandelaToRadiantIntensity;
using oka::kLuminousEfficacyD65;

inline size_t floatBlobOffset(size_t profileCount)
{
    return sizeof(IesBufferHeader) + profileCount * sizeof(IesProfileHeader);
}

template <typename ProfileType>
inline bool validProfile(const ProfileType& profile, size_t& sampleCount)
{
    const size_t nV = profile.verticalAngles.size();
    const size_t nH = profile.horizontalAngles.size();
    if (nV < 2 || nH < 1 || nV > std::numeric_limits<uint32_t>::max() || nH > std::numeric_limits<uint32_t>::max() ||
        nH > std::numeric_limits<size_t>::max() / nV)
    {
        return false;
    }

    sampleCount = nV * nH;
    if (profile.candela.size() < sampleCount)
    {
        return false;
    }

    const auto finiteNondecreasing = [](const std::vector<float>& values) {
        for (size_t i = 0; i < values.size(); ++i)
        {
            if (!std::isfinite(values[i]) || (i > 0 && values[i] < values[i - 1]))
            {
                return false;
            }
        }
        return true;
    };
    if (!finiteNondecreasing(profile.verticalAngles) || !finiteNondecreasing(profile.horizontalAngles))
    {
        return false;
    }
    for (size_t i = 0; i < sampleCount; ++i)
    {
        if (!std::isfinite(profile.candela[i]) || profile.candela[i] < 0.0f)
        {
            return false;
        }
    }
    return true;
}

/// Serialize the exact IES buffer consumed by both device backends. Invalid
/// profiles retain an empty header at their original index so packed light
/// indices cannot move to a neighbouring profile.
template <typename ProfileType>
inline std::vector<uint8_t> packProfiles(const std::vector<ProfileType>& profiles)
{
    constexpr size_t maxProfiles =
        (std::numeric_limits<uint32_t>::max() - sizeof(IesBufferHeader)) / sizeof(IesProfileHeader);
    if (profiles.size() > maxProfiles)
    {
        std::vector<uint8_t> empty(sizeof(IesBufferHeader), 0);
        return empty;
    }
    const size_t headerBytes = floatBlobOffset(profiles.size());

    std::vector<IesProfileHeader> headers(profiles.size());
    std::vector<float> floats;
    for (size_t i = 0; i < profiles.size(); ++i)
    {
        const ProfileType& profile = profiles[i];
        size_t sampleCount = 0;
        const size_t nV = profile.verticalAngles.size();
        const size_t nH = profile.horizontalAngles.size();
        if (!validProfile(profile, sampleCount))
        {
            continue;
        }
        const size_t packedNH = nH == 1 ? 2 : nH;
        const size_t packedSampleCount = nV * packedNH;
        const size_t angleCount = nV + packedNH;
        if (packedSampleCount > std::numeric_limits<uint32_t>::max() - angleCount ||
            floats.size() > std::numeric_limits<uint32_t>::max() - angleCount - packedSampleCount)
        {
            continue;
        }

        IesProfileHeader& header = headers[i];
        header.nVertical = static_cast<uint32_t>(nV);
        header.nHorizontal = static_cast<uint32_t>(packedNH);
        header.anglesOffset = static_cast<uint32_t>(floats.size());
        floats.insert(floats.end(), profile.verticalAngles.begin(), profile.verticalAngles.end());
        if (nH == 1)
        {
            floats.push_back(0.0f);
            floats.push_back(360.0f);
        }
        else
        {
            floats.insert(floats.end(), profile.horizontalAngles.begin(), profile.horizontalAngles.end());
        }
        header.candelaOffset = static_cast<uint32_t>(floats.size());

        float maximum = 0.0f;
        for (size_t c = 0; c < packedSampleCount; ++c)
        {
            const float value = profile.candela[c % sampleCount];
            maximum = std::max(maximum, value);
            floats.push_back(value * kCandelaToRadiantIntensity);
        }
        header.maxCandela = maximum * kCandelaToRadiantIntensity;
    }

    IesBufferHeader header{};
    header.profileCount = static_cast<uint32_t>(profiles.size());
    header.floatOffset = static_cast<uint32_t>(headerBytes);

    std::vector<uint8_t> bytes(headerBytes + floats.size() * sizeof(float), 0);
    std::memcpy(bytes.data(), &header, sizeof(header));
    if (!headers.empty())
    {
        std::memcpy(bytes.data() + sizeof(IesBufferHeader), headers.data(), headers.size() * sizeof(IesProfileHeader));
    }
    if (!floats.empty())
    {
        std::memcpy(bytes.data() + header.floatOffset, floats.data(), floats.size() * sizeof(float));
    }
    return bytes;
}

} // namespace oka::ies_pack
