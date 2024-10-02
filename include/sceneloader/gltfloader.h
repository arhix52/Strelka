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
    explicit GltfLoader(){}

    bool loadGltf(const std::string& modelPath, Scene& mScene);
};
} // namespace oka
