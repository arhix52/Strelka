#pragma once

#include <strelka/scene/scene.h>

#include <string>

namespace oka
{

/// Parse an IESNA LM-63 photometric file into a candela table.
/// Returns false on IO or format errors; `out` is left untouched then.
bool loadIesProfile(const std::string& path, Scene::IesProfile& out);

/// Bilinear sample of an IES profile. `dir` is the direction from the light
/// toward the shading point in the light's local frame, where −Z is the
/// photometric axis (same convention as every other Strelka light).
float sampleIesCandela(const Scene::IesProfile& profile, const glm::float3& localDir);

/// Unfold a symmetric azimuth table to the full turn and close it at 360.
///
/// loadIesProfile() calls this; it is exposed so the tests can drive it on a
/// hand-built profile. LM-63 encodes the symmetry in the last horizontal angle
/// -- 0 rotationally symmetric, 90 one quadrant, 180 one half, 360 the whole
/// turn -- and unfolding once at load time is what lets the cubic interpolation
/// in ies_math.h have real neighbours at the seam.
void unfoldIesAzimuth(Scene::IesProfile& profile);

} // namespace oka
