#pragma once

#include <strelka/scene/scene.h>

#include <string>

namespace oka
{

/// Write analytic lights + environment to <stem>_light.json (sidecar next to glTF).
bool saveLightsJson(const Scene& scene, const std::string& gltfOrJsonPath);

/// Load lights from an explicit *_light.json path into scene (clears existing lights first).
bool loadLightsJson(Scene& scene, const std::string& lightJsonPath);

/// Write scene graph / materials / meshes to .gltf/.glb via tinygltf.
/// Does not embed analytic lights (those stay in *_light.json).
bool saveGltf(const Scene& scene, const std::string& outputPath);

} // namespace oka
