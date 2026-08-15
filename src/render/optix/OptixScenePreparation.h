#pragma once

#include <cstdint>
#include <functional>

namespace oka
{

class Buffer;

namespace optix
{

/// The stages a scene's GPU-side build passes through, in the order they run.
///
/// This mirrors `oka::metal::BuildStage` deliberately -- the two backends must
/// make a loading scene look the same, and the editor's progress bar and its
/// "still building" gate read one interface for both. Ordered by dependency
/// first, and beyond that by how soon a stage produces something worth looking
/// at.
///
/// Environment comes second for the latter reason. It is the cheapest stage and
/// the only one that yields a complete, correct picture on its own -- an empty
/// top level means every ray misses and reaches the environment, so the frame is
/// the scene's own sky and the light it casts, with none of its objects in it
/// yet. The structures, which are the long pole, then build into that rather
/// than replacing a black screen.
enum class BuildStage : uint32_t
{
    Buffers = 0,      ///< vertices, indices, curve points and widths, skin data
    Environment,      ///< accumulation target, env map, lights, empty top level, SBT
    MaterialParams,   ///< the whole material table, with no textures in it yet
    Structures,       ///< every BLAS, then the real top level -- the long pole
    MaterialTextures, ///< maps, decoded into the live table as they arrive
    Tail,             ///< skinning setup, memory report
    Done,
};

/// Human-readable stage name. Also what the breadcrumb prints when a launch
/// faults during a build, so it is a function rather than a table nobody can
/// reach.
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

/// What fraction of the whole build each stage is worth.
///
/// Only used to turn a stage cursor into a bar position, and the weights are
/// rough on purpose: what matters is that the bar never goes backwards and that
/// the long pole occupies most of it. A bar that spends 90% of a fourteen-second
/// load sitting at 15% is worse than no bar, which is what equal weights give.
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

/// Fraction of the build already behind `stage`, in [0, 1].
inline float buildProgressBefore(BuildStage stage)
{
    float sum = 0.0f;
    for (uint32_t s = 0; s < static_cast<uint32_t>(stage); ++s)
    {
        sum += buildStageWeight(static_cast<BuildStage>(s));
    }
    return sum;
}

/// The work each stage does. Every hook is optional: an unset one is skipped and
/// the stage still advances, which is what lets a test drive the machine with
/// nothing behind it.
///
/// The two sliced stages return true when they are finished; the machine keeps
/// the cursor on them until they do.
struct SceneBuildHooks
{
    std::function<void()> onBuffersEnter;
    std::function<void()> buildBuffers;

    std::function<void()> onEnvironmentEnter;
    /// Accumulation target, environment map, light buffer, shader binding table,
    /// and an empty top level to trace against until real geometry replaces it.
    std::function<void(Buffer* output)> buildEnvironment;

    std::function<void()> onMaterialParamsEnter;
    std::function<void()> publishMaterialParams;

    std::function<void()> onStructuresEnter;
    std::function<bool(double /*budgetMs*/)> stepStructures;

    std::function<void()> onMaterialTexturesEnter;
    std::function<bool(double /*budgetMs*/)> stepMaterialTextures;

    std::function<void()> onTailEnter;
    std::function<void(Buffer* output)> buildTail;

    /// Called after every stage that did work, with the stage that ran and how
    /// long it took. The machine has no logger of its own so this header can be
    /// compiled by a test that links nothing.
    std::function<void(BuildStage, double /*elapsedMs*/)> onStageTimed;

    /// Wall clock in milliseconds. Injected so a test can drive the slice budget
    /// deterministically instead of racing a real clock.
    std::function<double()> nowMs;
};

class OptixScenePreparation
{
public:
    /// How much work one call of the sliced stages may do. Long enough that the
    /// per-slice overhead is noise, short enough that the window still answers
    /// the mouse -- one slice per displayed frame, at a frame that is not a fast
    /// one. The same 24 ms Metal uses, because the two backends are pacing the
    /// same editor loop.
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
    bool step(SceneBuildHooks& hooks, Buffer* output)
    {
        const BuildStage ran = mStage;
        const double started = hooks.nowMs ? hooks.nowMs() : 0.0;

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
            mStage = BuildStage::MaterialParams;
            break;

        case BuildStage::MaterialParams:
            if (hooks.onMaterialParamsEnter)
                hooks.onMaterialParamsEnter();
            if (hooks.publishMaterialParams)
                hooks.publishMaterialParams();
            mStage = BuildStage::Structures;
            break;

        case BuildStage::Structures:
            if (hooks.onStructuresEnter)
                hooks.onStructuresEnter();
            // The one stage too long to run whole. It stays current until it
            // says it is finished; an unset hook finishes immediately so the
            // machine cannot wedge on a backend that has nothing to build.
            if (!hooks.stepStructures || hooks.stepStructures(kBuildSliceMs))
            {
                mStage = BuildStage::MaterialTextures;
            }
            break;

        case BuildStage::MaterialTextures:
            if (hooks.onMaterialTexturesEnter)
                hooks.onMaterialTexturesEnter();
            if (!hooks.stepMaterialTextures || hooks.stepMaterialTextures(kBuildSliceMs))
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

        if (ran != BuildStage::Done && hooks.onStageTimed)
        {
            hooks.onStageTimed(ran, (hooks.nowMs ? hooks.nowMs() : 0.0) - started);
        }
        return mStage == BuildStage::Done;
    }

    /// Drive the build to completion in one call (CLI / renderSync).
    ///
    /// A synchronous caller wants the frame, not a responsive window, so nothing
    /// here is paced: the sliced stages are stepped until they report done.
    void finish(SceneBuildHooks& hooks, Buffer* output)
    {
        while (mStage != BuildStage::Done)
        {
            step(hooks, output);
        }
    }

private:
    BuildStage mStage = BuildStage::Done;
};

} // namespace optix
} // namespace oka
