#pragma once

#include <scene/scene.h>

#include <string>
#include <vector>

namespace oka
{

class GltfLoader
{
private:

public:
    explicit GltfLoader()

    bool loadGltf(const std::string& modelPath, Scene& mScene);

    void computeTangent(std::vector<Scene::Vertex>& _vertices,
                        const std::vector<uint32_t>& _indices) const;
};
} // namespace oka
