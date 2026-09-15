#pragma once

#include <host/scene_preparation.h>

namespace oka::metal
{

using scene_preparation::BuildStage;
using scene_preparation::buildStageName;
using scene_preparation::SceneBuildHooks;

class MetalScenePreparation
{
public:
    static constexpr double kBuildSliceMs = scene_preparation::ScenePreparation::kBuildSliceMs;

    void begin()
    {
        mPreparation.begin();
    }

    BuildStage stage() const
    {
        return mPreparation.stage();
    }
    bool isDone() const
    {
        return mPreparation.isDone();
    }
    bool isBuilding() const
    {
        return mPreparation.isBuilding();
    }

    bool step(SceneBuildHooks& hooks, Buffer* output);
    void finish(SceneBuildHooks& hooks, Buffer* output);

private:
    scene_preparation::ScenePreparation mPreparation;
};

} // namespace oka::metal
