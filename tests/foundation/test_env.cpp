#include <doctest/doctest.h>

#include <env.h>
#include <settings.h>

#include <cstdlib>

using oka::envBool;
using oka::envDouble;
using oka::envFlag;
using oka::envFloat;
using oka::envUint;

namespace
{

// setenv/unsetenv rather than putenv: putenv keeps the caller's buffer in the
// environment, and a test-local one goes out of scope while getenv still points
// at it.
struct ScopedEnv
{
    explicit ScopedEnv(const char* name, const char* value) : mName(name)
    {
        setenv(mName, value, 1);
    }
    ~ScopedEnv()
    {
        unsetenv(mName);
    }
    const char* mName;
};

} // namespace

TEST_CASE("env helpers fall back when the variable is absent or empty")
{
    unsetenv("STRELKA_TEST_ABSENT");
    CHECK(envFlag("STRELKA_TEST_ABSENT") == false);
    CHECK(envUint("STRELKA_TEST_ABSENT", 7) == 7);
    CHECK(envDouble("STRELKA_TEST_ABSENT", 1.5) == doctest::Approx(1.5));
    CHECK(envBool("STRELKA_TEST_ABSENT", true) == true);

    const ScopedEnv empty("STRELKA_TEST_EMPTY", "");
    // Present but empty: the flag is set, the value is not usable.
    CHECK(envFlag("STRELKA_TEST_EMPTY") == true);
    CHECK(envUint("STRELKA_TEST_EMPTY", 7) == 7);
}

TEST_CASE("env helpers parse well-formed values")
{
    const ScopedEnv n("STRELKA_TEST_NUM", "42");
    CHECK(envUint("STRELKA_TEST_NUM", 0) == 42);
    CHECK(envBool("STRELKA_TEST_NUM", false) == true);

    const ScopedEnv zero("STRELKA_TEST_ZERO", "0");
    CHECK(envUint("STRELKA_TEST_ZERO", 9) == 0);
    CHECK(envBool("STRELKA_TEST_ZERO", true) == false);

    const ScopedEnv f("STRELKA_TEST_FLOAT", "0.5");
    CHECK(envFloat("STRELKA_TEST_FLOAT", 1.0f) == doctest::Approx(0.5f));
}

// The point of the helpers: atoi() answered 0 here, and 0 is a meaningful
// setting for most of these knobs -- a typo would have been indistinguishable
// from asking for the feature to be off.
TEST_CASE("env helpers reject malformed values instead of reading them as zero")
{
    const ScopedEnv word("STRELKA_TEST_WORD", "yes");
    CHECK(envUint("STRELKA_TEST_WORD", 3) == 3);
    CHECK(envBool("STRELKA_TEST_WORD", true) == true);

    const ScopedEnv trailing("STRELKA_TEST_TRAILING", "12abc");
    CHECK(envUint("STRELKA_TEST_TRAILING", 3) == 3);

    const ScopedEnv negative("STRELKA_TEST_NEGATIVE", "-1");
    CHECK(envUint("STRELKA_TEST_NEGATIVE", 3) == 3);

    const ScopedEnv notANumber("STRELKA_TEST_NAN", "abc");
    CHECK(envDouble("STRELKA_TEST_NAN", 2.5) == doctest::Approx(2.5));
}

TEST_CASE("animation setting keys have one spelling")
{
    CHECK(oka::animationStateKey(0) == "render/animation/anim0/state");
    CHECK(oka::animationTimeKey(0) == "render/animation/anim0/time");
    CHECK(oka::animationStateKey(12) == "render/animation/anim12/state");

    // What the keys exist for: written under one index, read back under the
    // same one. eraseByPrefix() is how a scene change drops them, so they all
    // have to sit under the prefix it uses.
    oka::SettingsManager settings;
    settings.setAs<bool>(oka::animationStateKey(3), true);
    settings.setAs<float>(oka::animationTimeKey(3), 1.25f);
    CHECK(settings.getAs<bool>(oka::animationStateKey(3)) == true);
    CHECK(settings.getAs<float>(oka::animationTimeKey(3)) == doctest::Approx(1.25f));

    settings.eraseByPrefix("render/animation/anim");
    CHECK(settings.contains(oka::animationStateKey(3)) == false);
    CHECK(settings.contains(oka::animationTimeKey(3)) == false);
}
