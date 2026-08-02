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

} // namespace oka
