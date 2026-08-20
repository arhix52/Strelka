#include <doctest/doctest.h>

#include "OptixScenePreparation.h"

#include <string>
#include <vector>

using oka::optix::BuildStage;
using oka::optix::buildProgressBefore;
using oka::optix::buildStageName;
using oka::optix::OptixScenePreparation;
using oka::optix::SceneBuildHooks;

namespace
{
// Records the order stages ran in, and lets the two sliced stages be told how
// many calls they should take before reporting themselves finished. That is the
// whole contract the renderer depends on: the cursor must stay on a sliced stage
// until it says it is done, and must not run the next stage in the same call.
struct Recorder
{
    std::vector<std::string> ran;
    int structureSlices = 1;
    int textureSlices = 1;
    int structureCalls = 0;
    int textureCalls = 0;
    double lastStructureBudget = -1.0;

    SceneBuildHooks hooks()
    {
        SceneBuildHooks h;
        h.buildBuffers = [this]() { ran.emplace_back("buffers"); };
        h.buildEnvironment = [this](oka::Buffer*) { ran.emplace_back("environment"); };
        h.publishMaterialParams = [this]() { ran.emplace_back("material params"); };
        h.stepStructures = [this](double budgetMs) {
            ran.emplace_back("structures");
            lastStructureBudget = budgetMs;
            return ++structureCalls >= structureSlices;
        };
        h.stepMaterialTextures = [this](double) {
            ran.emplace_back("material textures");
            return ++textureCalls >= textureSlices;
        };
        h.buildTail = [this](oka::Buffer*) { ran.emplace_back("tail"); };
        return h;
    }
};
} // namespace

// The order is not cosmetic. Environment runs second so a scene has its sky and
// the light it casts on screen while the structures -- the long pole -- build
// into it; it used to run last, which is why a large scene showed nothing at all
// until every last byte of it was resident.
TEST_CASE("the environment is ready before the structures start")
{
    Recorder rec;
    SceneBuildHooks hooks = rec.hooks();
    OptixScenePreparation prep;
    prep.begin();
    prep.finish(hooks, nullptr);

    const std::vector<std::string> expected = { "buffers",    "environment",       "material params",
                                                "structures", "material textures", "tail" };
    CHECK(rec.ran == expected);
    CHECK(prep.isDone());
}

TEST_CASE("a machine that has not begun is not building")
{
    OptixScenePreparation prep;
    CHECK(prep.isDone());
    CHECK_FALSE(prep.isBuilding());

    // A step on a finished machine must be inert, because render() calls it on
    // every frame of an already-loaded scene.
    Recorder rec;
    SceneBuildHooks hooks = rec.hooks();
    CHECK(prep.step(hooks, nullptr));
    CHECK(rec.ran.empty());
}

TEST_CASE("one stage per step, and a sliced stage keeps the cursor")
{
    Recorder rec;
    rec.structureSlices = 3;
    SceneBuildHooks hooks = rec.hooks();
    OptixScenePreparation prep;
    prep.begin();

    CHECK_FALSE(prep.step(hooks, nullptr)); // buffers
    CHECK(prep.stage() == BuildStage::Environment);
    CHECK_FALSE(prep.step(hooks, nullptr)); // environment
    CHECK_FALSE(prep.step(hooks, nullptr)); // material params
    CHECK(prep.stage() == BuildStage::Structures);

    // Two slices that do not finish leave the cursor where it is, and crucially
    // do not run the textures stage early.
    CHECK_FALSE(prep.step(hooks, nullptr));
    CHECK(prep.stage() == BuildStage::Structures);
    CHECK_FALSE(prep.step(hooks, nullptr));
    CHECK(prep.stage() == BuildStage::Structures);
    CHECK(rec.textureCalls == 0);

    CHECK_FALSE(prep.step(hooks, nullptr)); // third slice finishes
    CHECK(prep.stage() == BuildStage::MaterialTextures);
    CHECK(rec.structureCalls == 3);
}

TEST_CASE("the slice budget the stages are handed is the published one")
{
    Recorder rec;
    SceneBuildHooks hooks = rec.hooks();
    OptixScenePreparation prep;
    prep.begin();
    prep.finish(hooks, nullptr);
    CHECK(rec.lastStructureBudget == doctest::Approx(OptixScenePreparation::kBuildSliceMs));
}

// finish() is what renderSync and the CLI take, and it must not be able to spin
// forever on a stage that keeps asking for another slice.
TEST_CASE("finish drives a many-sliced build all the way to done")
{
    Recorder rec;
    rec.structureSlices = 17;
    rec.textureSlices = 9;
    SceneBuildHooks hooks = rec.hooks();
    OptixScenePreparation prep;
    prep.begin();
    prep.finish(hooks, nullptr);

    CHECK(prep.isDone());
    CHECK(rec.structureCalls == 17);
    CHECK(rec.textureCalls == 9);
    CHECK(rec.ran.back() == "tail");
}

// A backend with nothing to build must not wedge: an unset sliced hook has to
// read as "finished", not as "ask again forever".
TEST_CASE("unset hooks do not stall the build")
{
    SceneBuildHooks empty;
    OptixScenePreparation prep;
    prep.begin();
    for (int i = 0; i < 16 && !prep.isDone(); ++i)
    {
        prep.step(empty, nullptr);
    }
    CHECK(prep.isDone());
}

// The bar is drawn from these, so they have to be monotonic and to end at one.
// A bar that goes backwards mid-load is worse than no bar.
TEST_CASE("build progress never goes backwards and reaches one")
{
    float previous = -1.0f;
    for (uint32_t s = 0; s <= static_cast<uint32_t>(BuildStage::Done); ++s)
    {
        const float p = buildProgressBefore(static_cast<BuildStage>(s));
        CHECK(p >= previous);
        previous = p;
    }
    CHECK(buildProgressBefore(BuildStage::Done) == doctest::Approx(1.0f));
    CHECK(buildProgressBefore(BuildStage::Buffers) == doctest::Approx(0.0f));
}

TEST_CASE("every stage has a name")
{
    for (uint32_t s = 0; s <= static_cast<uint32_t>(BuildStage::Done); ++s)
    {
        CHECK(std::string(buildStageName(static_cast<BuildStage>(s))) != "unknown");
    }
}
