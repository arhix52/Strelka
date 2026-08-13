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
// Ordered by dependency, and beyond that by how soon each stage produces
// something worth looking at. Environment comes second for the latter reason:
// it is the cheapest stage and the only one that yields a complete, correct
// picture on its own -- the sky, and the light it casts -- so the scene has
// something on screen while the structures, which are the long pole, build into
// it. It used to run last, which is why a scene showed nothing until all of it
// was ready.
enum class BuildStage : uint32_t
{
    Buffers = 0,
    Environment,     ///< accumulation target, env map, empty top level
    MaterialParams,  ///< the whole material table, with no textures in it yet
    Structures,      ///< the long pole
    MaterialTextures,///< maps, filled into the live table as they decode
    Tail,            ///< skinning, memory report
    Done,
};

struct SceneBuildHooks
{
    // Entering Buffers: reset temporal state, begin Geometry progress.
    std::function<void()> onBuffersEnter;
    std::function<void()> buildBuffers;

    std::function<void()> onEnvironmentEnter;
    /// Accumulation target, environment map, and an empty top level to trace
    /// against until real geometry replaces it.
    std::function<void(Buffer* output)> buildEnvironment;

    // The material table without its maps: cheap, and everything the structures
    // stage needs to know about materials comes out of it.
    std::function<void()> onMaterialParamsEnter;
    std::function<void()> publishMaterialParams;

    // The maps. Runs after the structures so geometry is on screen, in flat
    // material colours, while they decode.
    std::function<void()> onMaterialTexturesEnter;
    std::function<bool(double /*budgetMs*/)> stepMaterialTextures;

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
