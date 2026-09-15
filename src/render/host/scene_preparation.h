#pragma once

#include <cstdint>
#include <functional>

namespace oka
{

class Buffer;

namespace scene_preparation
{

enum class BuildStage : uint32_t
{
    Buffers = 0,
    Environment,
    MaterialParams,
    Structures,
    MaterialTextures,
    Tail,
    Done,
};

inline const char* buildStageName(BuildStage stage)
{
    switch (stage)
    {
    case BuildStage::Buffers:
        return "buffers";
    case BuildStage::Environment:
        return "environment";
    case BuildStage::MaterialParams:
        return "material params";
    case BuildStage::Structures:
        return "structures";
    case BuildStage::MaterialTextures:
        return "material textures";
    case BuildStage::Tail:
        return "tail";
    case BuildStage::Done:
        return "done";
    }
    return "unknown";
}

inline float buildStageWeight(BuildStage stage)
{
    switch (stage)
    {
    case BuildStage::Buffers:
        return 0.10f;
    case BuildStage::Environment:
        return 0.05f;
    case BuildStage::MaterialParams:
        return 0.02f;
    case BuildStage::Structures:
        return 0.53f;
    case BuildStage::MaterialTextures:
        return 0.28f;
    case BuildStage::Tail:
        return 0.02f;
    case BuildStage::Done:
        return 0.0f;
    }
    return 0.0f;
}

inline float buildProgressBefore(BuildStage stage)
{
    float sum = 0.0f;
    for (uint32_t value = 0; value < static_cast<uint32_t>(stage); ++value)
    {
        sum += buildStageWeight(static_cast<BuildStage>(value));
    }
    return sum;
}

struct SceneBuildHooks
{
    std::function<void()> onBuffersEnter;
    std::function<void()> buildBuffers;
    std::function<void()> onEnvironmentEnter;
    std::function<void(Buffer*)> buildEnvironment;
    std::function<void()> onMaterialParamsEnter;
    std::function<void()> publishMaterialParams;
    std::function<void()> onStructuresEnter;
    std::function<bool(double)> stepStructures;
    std::function<void()> onMaterialTexturesEnter;
    std::function<bool(double)> stepMaterialTextures;
    std::function<void()> onTailEnter;
    std::function<void(Buffer*)> buildTail;
    std::function<void(BuildStage, double)> onStageTimed;
    std::function<double()> nowMs;
};

class ScenePreparation
{
public:
    static constexpr double kBuildSliceMs = 24.0;

    void begin()
    {
        mStage = BuildStage::Buffers;
    }

    BuildStage stage() const
    {
        return mStage;
    }

    bool isDone() const
    {
        return mStage == BuildStage::Done;
    }

    bool isBuilding() const
    {
        return !isDone();
    }

    bool step(SceneBuildHooks& hooks, Buffer* output)
    {
        const BuildStage ran = mStage;
        const double started = hooks.nowMs ? hooks.nowMs() : 0.0;

        switch (mStage)
        {
        case BuildStage::Buffers:
            invoke(hooks.onBuffersEnter);
            invoke(hooks.buildBuffers);
            mStage = BuildStage::Environment;
            break;
        case BuildStage::Environment:
            invoke(hooks.onEnvironmentEnter);
            if (hooks.buildEnvironment)
            {
                hooks.buildEnvironment(output);
            }
            mStage = BuildStage::MaterialParams;
            break;
        case BuildStage::MaterialParams:
            invoke(hooks.onMaterialParamsEnter);
            invoke(hooks.publishMaterialParams);
            mStage = BuildStage::Structures;
            break;
        case BuildStage::Structures:
            invoke(hooks.onStructuresEnter);
            if (!hooks.stepStructures || hooks.stepStructures(kBuildSliceMs))
            {
                mStage = BuildStage::MaterialTextures;
            }
            break;
        case BuildStage::MaterialTextures:
            invoke(hooks.onMaterialTexturesEnter);
            if (!hooks.stepMaterialTextures || hooks.stepMaterialTextures(kBuildSliceMs))
            {
                mStage = BuildStage::Tail;
            }
            break;
        case BuildStage::Tail:
            invoke(hooks.onTailEnter);
            if (hooks.buildTail)
            {
                hooks.buildTail(output);
            }
            mStage = BuildStage::Done;
            break;
        case BuildStage::Done:
            break;
        }

        if (ran != BuildStage::Done && hooks.onStageTimed)
        {
            hooks.onStageTimed(ran, (hooks.nowMs ? hooks.nowMs() : 0.0) - started);
        }
        return isDone();
    }

    void finish(SceneBuildHooks& hooks, Buffer* output)
    {
        while (!isDone())
        {
            step(hooks, output);
        }
    }

private:
    static void invoke(const std::function<void()>& function)
    {
        if (function)
        {
            function();
        }
    }

    BuildStage mStage = BuildStage::Done;
};

} // namespace scene_preparation
} // namespace oka
