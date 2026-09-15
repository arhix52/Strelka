#include <doctest/doctest.h>

#include "host/scene_preparation.h"

#include <string>
#include <vector>

using oka::scene_preparation::buildProgressBefore;
using oka::scene_preparation::BuildStage;
using oka::scene_preparation::buildStageName;
using oka::scene_preparation::SceneBuildHooks;
using oka::scene_preparation::ScenePreparation;

namespace
{
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

TEST_CASE("the environment is ready before the structures start")
{
    Recorder rec;
    SceneBuildHooks hooks = rec.hooks();
    ScenePreparation prep;
    prep.begin();
    prep.finish(hooks, nullptr);

    const std::vector<std::string> expected = { "buffers",    "environment",       "material params",
                                                "structures", "material textures", "tail" };
    CHECK(rec.ran == expected);
    CHECK(prep.isDone());
}

TEST_CASE("a machine that has not begun is not building")
{
    ScenePreparation prep;
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
    ScenePreparation prep;
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
    ScenePreparation prep;
    prep.begin();
    prep.finish(hooks, nullptr);
    CHECK(rec.lastStructureBudget == doctest::Approx(ScenePreparation::kBuildSliceMs));
}

// finish() is what renderSync and the CLI take, and it must not be able to spin
// forever on a stage that keeps asking for another slice.
TEST_CASE("finish drives a many-sliced build all the way to done")
{
    Recorder rec;
    rec.structureSlices = 17;
    rec.textureSlices = 9;
    SceneBuildHooks hooks = rec.hooks();
    ScenePreparation prep;
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
    ScenePreparation prep;
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
