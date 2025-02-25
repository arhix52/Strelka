#pragma once

#include <scene/scene.h>

#include <string>

namespace oka
{

class GltfLoader
{
private:

public:
    explicit GltfLoader(){}

    bool loadGltf(const std::string& modelPath, Scene& mScene);
};
} // namespace oka
