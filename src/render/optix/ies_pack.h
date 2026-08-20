#pragma once

#include <strelka/scene/light_desc.h>

// Packing IESNA LM-63 candela tables into the single flat buffer the shading
// path reads, and the one unit conversion that goes with it.
//
// The layout is a header, one profile header per profile, then a float blob:
//
//   IesBufferHeader                       // profileCount, floatOffset
//   IesProfileHeader[profileCount]        // grid sizes and blob indices
//   float[]                               // per profile: vertical angles,
//                                         // horizontal angles, then candela
//
// One buffer rather than one per light because a light's profile is chosen by
// an index it carries in points[0].y, and an index into an array of pointers
// would need a second indirection on the device for nothing.
//
// The two header structs are mirrored here, rather than included from
// <lights.h>, so that this file and its tests need no CUDA. OptixRender.cpp
// static_asserts each against the device-side struct, so a divergence is a
// build failure rather than a buffer the shader reads at the wrong stride.
// They are field-for-field the Metal ShaderTypes.h pair as well.

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>


namespace oka::optix_ies
{

struct IesBufferHeader
{
    uint32_t profileCount;
    uint32_t floatOffset; // byte offset of the float blob from the buffer start
    uint32_t pad0;
    uint32_t pad1;
};

struct IesProfileHeader
{
    uint32_t nVertical;
    uint32_t nHorizontal;
    uint32_t anglesOffset;  // index into the float blob: vertical then horizontal
    uint32_t candelaOffset; // index into the float blob
    float maxCandela;
    float pad0;
    float pad1;
    float pad2;
};

/// Luminous efficacy used to turn a photometric table into radiometric units.
///
// Re-exported into this namespace so callers that already say
// oka::optix_ies::kCandelaToRadiantIntensity keep working; the definition lives
// in scene/light_desc.h, next to the other photometric conversions.
using oka::kCandelaToRadiantIntensity;
using oka::kLuminousEfficacyD65;

/// An IES file is in candela -- lumens per steradian -- while a light's colour
/// in this renderer is radiant intensity, watts per steradian. Converting needs
/// a luminous efficacy, and a photometric file carries no spectrum of its own,
/// so a standard illuminant has to be assumed. See kLuminousEfficacyD65 in
/// scene/light_desc.h for which one and why, including why the glTF loader's
/// 683 is a different number for a different job.

/// One profile as the scene holds it. Mirrors Scene::IesProfile's numeric
/// fields; the path is not needed to pack.
struct Profile
{
    std::vector<float> verticalAngles;   // degrees, ascending
    std::vector<float> horizontalAngles; // degrees, ascending
    std::vector<float> candela;          // row-major: v + h * nVertical
    float maxCandela = 0.0f;
};

/// Byte size of the header block for `profileCount` profiles, which is also the
/// offset of the float blob.
inline size_t floatBlobOffset(size_t profileCount)
{
    return sizeof(IesBufferHeader) + profileCount * sizeof(IesProfileHeader);
}

/// Pack profiles into the flat device buffer.
///
/// Always returns at least a header: a zero profile count is a valid empty
/// table, and handing the shading path a real pointer to one is what lets a
/// scene with no IES light take the same code path as a scene with one, rather
/// than needing a null check per connection.
///
/// A profile whose grid is degenerate -- fewer than two vertical angles, no
/// horizontal angle, or a candela table shorter than the grid it declares -- is
/// packed as an empty entry rather than dropped. Dropping one would shift every
/// later profile's index, and those indices are already baked into the light
/// records by the time this runs.
inline std::vector<uint8_t> packProfiles(const std::vector<Profile>& profiles)
{
    std::vector<IesProfileHeader> headers(profiles.size());
    std::vector<float> floats;

    for (size_t i = 0; i < profiles.size(); ++i)
    {
        const Profile& p = profiles[i];
        IesProfileHeader& h = headers[i];
        h = IesProfileHeader{};

        const size_t nV = p.verticalAngles.size();
        const size_t nH = p.horizontalAngles.size();
        if (nV < 2 || nH < 1 || p.candela.size() < nV * nH)
        {
            continue;
        }

        h.nVertical = (uint32_t)nV;
        h.nHorizontal = (uint32_t)nH;
        h.anglesOffset = (uint32_t)floats.size();
        floats.insert(floats.end(), p.verticalAngles.begin(), p.verticalAngles.end());
        floats.insert(floats.end(), p.horizontalAngles.begin(), p.horizontalAngles.end());
        h.candelaOffset = (uint32_t)floats.size();
        for (size_t c = 0; c < nV * nH; ++c)
        {
            floats.push_back(p.candela[c] * kCandelaToRadiantIntensity);
        }
        h.maxCandela = p.maxCandela * kCandelaToRadiantIntensity;
    }

    IesBufferHeader header{};
    header.profileCount = (uint32_t)profiles.size();
    header.floatOffset = (uint32_t)floatBlobOffset(profiles.size());

    std::vector<uint8_t> bytes(header.floatOffset + floats.size() * sizeof(float), 0);
    std::memcpy(bytes.data(), &header, sizeof(header));
    if (!headers.empty())
    {
        std::memcpy(bytes.data() + sizeof(IesBufferHeader), headers.data(),
                    headers.size() * sizeof(IesProfileHeader));
    }
    if (!floats.empty())
    {
        std::memcpy(bytes.data() + header.floatOffset, floats.data(), floats.size() * sizeof(float));
    }
    return bytes;
}

} // namespace oka::optix_ies

