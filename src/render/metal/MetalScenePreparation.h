#pragma once

#include <strelka/render/buffer.h>

#include <cstdint>
#include <functional>

namespace oka
{
namespace metal
{

// Thin BuildStage state machine for primary scene preparation.
// Owns only the stage cursor and slice budget — GPU resources stay in domains.
enum class BuildStage : uint32_t
{
    Buffers = 0,
    Materials,
    Structures,
    Tail, ///< accumulation buffer, skinning, environment
    Done,
};

struct SceneBuildHooks
{
    // Entering Buffers: reset temporal state, begin Geometry progress.
    std::function<void()> onBuffersEnter;
    std::function<void()> buildBuffers;

    // Materials progress begin when build not yet active.
    std::function<void()> onMaterialsEnter;
    std::function<bool(double /*budgetMs*/)> stepMaterials;

    std::function<void()> onStructuresEnter;
    // Called once when Structures begins: force static BLAS for load.
    std::function<void()> onStructuresBegin;
    std::function<bool(double /*budgetMs*/)> stepStructures;

    std::function<void()> onTailEnter;
    std::function<void(Buffer* output)> buildTail;
};

class MetalScenePreparation
{
public:
    // How much work one call of the sliced stages may do. Long enough that the
    // per-slice overhead is noise, short enough that the window still answers the
    // mouse -- one slice per displayed frame, at a frame that is not a fast one.
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
        return mStage != BuildStage::Done;
    }

    /// Runs the current stage and moves to the next. Returns true once finished.
    bool step(SceneBuildHooks& hooks, Buffer* output);
    /// Drive the build to completion in one call (CLI / renderSync).
    void finish(SceneBuildHooks& hooks, Buffer* output);

private:
    BuildStage mStage = BuildStage::Done;
};

} // namespace metal
} // namespace oka
