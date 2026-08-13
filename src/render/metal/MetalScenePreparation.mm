#include "MetalScenePreparation.h"

#include <log.h>

#include <chrono>

#include <Metal/Metal.hpp>

namespace oka
{
namespace metal
{

bool MetalScenePreparation::step(SceneBuildHooks& hooks, Buffer* output)
{
    NS::AutoreleasePool* pPool = NS::AutoreleasePool::alloc()->init();
    const auto started = std::chrono::steady_clock::now();
    const BuildStage ran = mStage;

    switch (mStage)
    {
    case BuildStage::Buffers:
        if (hooks.onBuffersEnter)
            hooks.onBuffersEnter();
        if (hooks.buildBuffers)
            hooks.buildBuffers();
        mStage = BuildStage::Environment;
        break;

    case BuildStage::Environment:
        if (hooks.onEnvironmentEnter)
            hooks.onEnvironmentEnter();
        if (hooks.buildEnvironment)
            hooks.buildEnvironment(output);
        mStage = BuildStage::Materials;
        break;

    case BuildStage::Materials:
        if (hooks.onMaterialsEnter)
            hooks.onMaterialsEnter();
        if (hooks.stepMaterials && hooks.stepMaterials(kBuildSliceMs))
        {
            mStage = BuildStage::Structures;
        }
        break;

    case BuildStage::Structures:
        if (hooks.onStructuresEnter)
            hooks.onStructuresEnter();
        if (hooks.onStructuresBegin)
            hooks.onStructuresBegin();
        // The one stage too long to run whole: two seconds on the pine forest,
        // against roughly one frame's worth per slice here. The stage stays
        // current until the build says it is finished.
        if (hooks.stepStructures && hooks.stepStructures(kBuildSliceMs))
        {
            mStage = BuildStage::Tail;
        }
        break;

    case BuildStage::Tail:
        if (hooks.onTailEnter)
            hooks.onTailEnter();
        if (hooks.buildTail)
            hooks.buildTail(output);
        mStage = BuildStage::Done;
        break;

    case BuildStage::Done:
        break;
    }

    if (ran != BuildStage::Done)
    {
        static const char* kStageNames[] = { "buffers", "environment", "materials", "structures", "tail" };
        STRELKA_DEBUG("Scene build stage '{}' took {:.0f} ms", kStageNames[(uint32_t)ran],
                      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count());
    }
    pPool->release();
    return mStage == BuildStage::Done;
}

void MetalScenePreparation::finish(SceneBuildHooks& hooks, Buffer* output)
{
    while (mStage != BuildStage::Done)
    {
        step(hooks, output);
    }
}

} // namespace metal
} // namespace oka
