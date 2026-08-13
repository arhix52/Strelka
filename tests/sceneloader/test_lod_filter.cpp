#include <doctest/doctest.h>

#include <strelka/sceneloader/lod_filter.h>

using oka::isProxyOrLowerLod;

// The rule decides whether a node's geometry is drawn at all, so a false
// positive silently deletes an object from the scene and a false negative puts
// a flat untextured hull back on top of the detailed mesh. Both failures are
// invisible in aggregate metrics, which is why the cases below are spelled out
// one at a time rather than sampled.

TEST_CASE("LOD filter keeps level zero and drops the alternatives")
{
    // The names that actually occur in the pine forest export.
    CHECK_FALSE(isProxyOrLowerLod("cover_01_lod0"));
    CHECK(isProxyOrLowerLod("cover_01_lod1"));
    CHECK_FALSE(isProxyOrLowerLod("dead_tree_lod0"));
    CHECK(isProxyOrLowerLod("dead_tree_lod1"));

    // Level zero written with a separator is still level zero.
    CHECK_FALSE(isProxyOrLowerLod("tree_trunk_lod_0"));
    CHECK(isProxyOrLowerLod("tree_trunk_lod_1"));
}

TEST_CASE("LOD filter drops proxies whatever level they claim")
{
    // A proxy is a stand-in for geometry that is drawn in full elsewhere, so it
    // goes regardless of the level suffix -- including lod0, which the level
    // rule on its own would keep.
    CHECK(isProxyOrLowerLod("rock_proxy"));
    CHECK(isProxyOrLowerLod("rock_proxy_lod1"));
    CHECK(isProxyOrLowerLod("dead_tree_proxy_lod0"));
    CHECK(isProxyOrLowerLod("branch_proxy"));
    CHECK(isProxyOrLowerLod("proxy_bark"));
    CHECK(isProxyOrLowerLod("PROXY_LEAVES_FIR"));
}

TEST_CASE("LOD filter is not fooled by numbers that are not levels")
{
    // Two digits, and the first is not zero: lod10 is an alternative, not the
    // base level. Parsing only the first digit would keep it.
    CHECK(isProxyOrLowerLod("rock_lod10"));
    CHECK(isProxyOrLowerLod("rock_lod12"));
    // ...and a padded zero is still zero, however many digits it is written in.
    CHECK_FALSE(isProxyOrLowerLod("rock_lod00"));
    CHECK_FALSE(isProxyOrLowerLod("rock_lod000"));
    // A level number longer than any integer type must not decide the answer by
    // overflowing. It is non-zero, so it is an alternative.
    CHECK(isProxyOrLowerLod("rock_lod99999999999999999999999999"));
}

TEST_CASE("LOD filter leaves ordinary names alone")
{
    // "lod" appears inside real words. Matching it without a following number
    // would delete the object.
    CHECK_FALSE(isProxyOrLowerLod("lodge"));
    CHECK_FALSE(isProxyOrLowerLod("lodge_roof"));
    CHECK_FALSE(isProxyOrLowerLod("melody"));
    CHECK_FALSE(isProxyOrLowerLod("lod"));
    CHECK_FALSE(isProxyOrLowerLod("lod_"));
    // Nothing to do with levels of detail at all.
    CHECK_FALSE(isProxyOrLowerLod("rock_moss_set_01"));
    CHECK_FALSE(isProxyOrLowerLod("Camera1"));
    CHECK_FALSE(isProxyOrLowerLod(""));
}

TEST_CASE("LOD filter ignores case, as exporters do not agree on it")
{
    CHECK(isProxyOrLowerLod("Rock_LOD1"));
    CHECK(isProxyOrLowerLod("ROCK_LOD_2"));
    CHECK(isProxyOrLowerLod("Rock_Proxy"));
    CHECK_FALSE(isProxyOrLowerLod("Rock_LOD0"));
}

TEST_CASE("LOD filter reads the level next to the marker, not elsewhere")
{
    // A number earlier in the name is part of the object's own name; only the
    // digits attached to "lod" say which level this is.
    CHECK_FALSE(isProxyOrLowerLod("cover_01_lod0"));
    CHECK_FALSE(isProxyOrLowerLod("set_12_piece_3_lod0"));
    CHECK(isProxyOrLowerLod("set_12_piece_3_lod4"));

    // More than one marker: the object is an alternative if any of them says so,
    // which is what a name like this one means in practice.
    CHECK(isProxyOrLowerLod("lod0_group_lod2"));
    CHECK_FALSE(isProxyOrLowerLod("lod0_group_lod0"));
}
