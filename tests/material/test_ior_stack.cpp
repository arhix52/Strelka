#include <doctest/doctest.h>

#include <strelka/material/material_math.h>
#include <strelka/material/ior_stack.h>

// ---------------------------------------------------------------------------
// The nested-dielectric stack.
//
// A path inside glass carries what it is inside of, so that the far wall knows
// which index ratio to refract by and the segment before it knows whose
// absorption to apply. Entering pushes, leaving pops -- and "leaving" has to
// remove the entry for the surface actually being left.
//
// Priority cannot do that on its own. It answers a different question: which
// dielectric wins where two of them overlap. glTF has no way to author it, so
// the loader gives one value to everything transmissive -- the Isometric
// Bathroom has twelve materials all at 10, from the bath water to the lotion in
// a bottle to the shower glass. Popping by priority in that scene removes
// whichever of them is topmost, which is a different object almost every time.
// ---------------------------------------------------------------------------

namespace
{

constexpr unsigned int kShared = 10; // what the loader gives every transmissive

IorStack with(unsigned int m0, float ior0, unsigned int m1, float ior1)
{
    IorStack s{};
    ior_stack_init(s);
    ior_stack_push(s, kShared, ior0, m0);
    ior_stack_push(s, kShared, ior1, m1);
    return s;
}

} // namespace

TEST_CASE("leaving one of two equal-priority media removes that one")
{
    // Enter the water, then a bubble inside it, then leave the *water* -- which
    // is what a ray does when it crosses the tub's far wall while a bubble is
    // still between it and the camera. The bubble has to survive.
    IorStack s = with(/*water*/ 3u, 1.33f, /*bubble*/ 7u, 1.60f);
    REQUIRE(s.top == 1);

    ior_stack_pop(s, kShared, /*water*/ 3u);

    CHECK(s.top == 0);
    CHECK(ior_entry_material(s.entries[0]) == 7u);
    CHECK(ior_stack_current_material(s) == 7u);
    CHECK(ior_stack_current_ior(s) == doctest::Approx(1.60f));
}

TEST_CASE("leaving the innermost medium still works")
{
    IorStack s = with(3u, 1.33f, 7u, 1.60f);
    ior_stack_pop(s, kShared, 7u);

    CHECK(s.top == 0);
    CHECK(ior_stack_current_material(s) == 3u);
    CHECK(ior_stack_current_ior(s) == doctest::Approx(1.33f));
}

TEST_CASE("popping by priority alone would take the wrong one")
{
    // The behaviour this replaces, stated so the test says what it is for: with
    // both entries at the same priority a top-down priority search finds the
    // bubble, and a ray leaving the water would come out believing it is inside
    // a bubble at IOR 1.6 rather than in air.
    const IorStack s = with(3u, 1.33f, 7u, 1.60f);
    int topmost_by_priority = -1;
    for (int i = s.top; i >= 0; --i)
    {
        if (ior_entry_priority(s.entries[i]) == kShared)
        {
            topmost_by_priority = i;
            break;
        }
    }
    REQUIRE(topmost_by_priority == 1);
    CHECK(ior_entry_material(s.entries[topmost_by_priority]) != 3u);
}

TEST_CASE("peeking the exterior IOR identifies the material among equal priorities")
{
    const IorStack s = with(/*water*/ 3u, 1.33f, /*bubble*/ 7u, 1.60f);

    CHECK(ior_stack_peek_after_pop_material(s, /*water*/ 3u) == doctest::Approx(1.60f));
    CHECK(ior_stack_peek_after_pop_material(s, /*bubble*/ 7u) == doctest::Approx(1.33f));
}

TEST_CASE("an exact material pop preserves an equal-priority enclosing medium")
{
    IorStack s = with(/*water*/ 3u, 1.33f, /*bubble*/ 7u, 1.60f);

    CHECK_FALSE(ior_stack_has_material(s, /*never entered*/ 99u));
    CHECK(ior_stack_pop_material(s, /*never entered*/ 99u) == doctest::Approx(1.60f));
    CHECK(s.top == 1);
    CHECK(ior_stack_current_material(s) == 7u);
}

TEST_CASE("an exit with nothing on the stack leaves it empty rather than negative")
{
    IorStack s{};
    ior_stack_init(s);
    const float ior = ior_stack_pop(s, kShared, 3u);
    CHECK(s.top == -1);
    CHECK(ior == doctest::Approx(1.0f));
    CHECK(ior_stack_current_material(s) == 0xFFFFFFFFu);
}

TEST_CASE("an exit for a material never entered falls back to the priority")
{
    // The case the priority search was written for, and the one an open mesh
    // produces: the path is inside something it has no record of entering.
    // Removing an equal-priority entry is the best available answer and is what
    // keeps a stack from growing without bound.
    IorStack s{};
    ior_stack_init(s);
    ior_stack_push(s, kShared, 1.5f, /*glass*/ 4u);

    ior_stack_pop(s, kShared, /*never entered*/ 99u);

    CHECK(s.top == -1);
    CHECK(ior_stack_current_ior(s) == doctest::Approx(1.0f));
}

TEST_CASE("a full stack drops the innermost push rather than corrupting itself")
{
    // IOR_STACK_SIZE is 4 and push is silent when it is full. An open mesh can
    // reach that, so what matters is that the stack stays consistent.
    IorStack s{};
    ior_stack_init(s);
    for (unsigned int i = 0; i < IOR_STACK_SIZE + 2u; ++i)
    {
        ior_stack_push(s, kShared, 1.1f + 0.1f * (float)i, i);
    }
    CHECK(s.top == IOR_STACK_SIZE - 1);
    CHECK(ior_stack_current_material(s) == IOR_STACK_SIZE - 1u);
}

// The stack is carried per path on both backends -- inside OptiX's PerRayData,
// where every byte is a byte of per-thread continuation stack, and in Metal's
// per-pixel side table, which is sized from sizeof(IorStack) at run time. The
// packing that makes an entry eight bytes instead of twelve is therefore a
// layout contract, not an implementation detail, and these are the two things it
// has to keep true.
TEST_CASE("packed entries are eight bytes and round-trip both fields")
{
    CHECK(sizeof(IorStackEntry) == 8u);
    CHECK(sizeof(IorStack) == size_t{ 8 } * IOR_STACK_SIZE + sizeof(int));

    IorStack s{};
    ior_stack_init(s);
    // The extremes of each field: the largest priority the top eight bits hold
    // and the largest material index the bottom twenty-four do.
    ior_stack_push(s, 255u, 1.42f, IOR_ENTRY_MATERIAL_MASK);
    CHECK(ior_entry_priority(s.entries[0]) == 255u);
    CHECK(ior_entry_material(s.entries[0]) == IOR_ENTRY_MATERIAL_MASK);
    CHECK(s.entries[0].ior == doctest::Approx(1.42f));

    // A priority does not bleed into the material index it shares a word with.
    ior_stack_push(s, 10u, 1.33f, 7u);
    CHECK(ior_entry_priority(s.entries[1]) == 10u);
    CHECK(ior_entry_material(s.entries[1]) == 7u);
}
