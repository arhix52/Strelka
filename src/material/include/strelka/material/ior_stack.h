#ifndef STRELKA_IOR_STACK_H
#define STRELKA_IOR_STACK_H

// ============================================================================
// ior_stack.h -- Priority-based IOR stack for nested dielectrics
//
// Each dielectric material has a priority (0 = air, higher = denser medium).
// The ray carries a small stack of (priority, ior) pairs. When entering a
// dielectric, we push; when exiting, we pop the matching priority.
// The current IOR is the top-of-stack entry (or 1.0 for air if empty).
// ============================================================================

#include "material_math.h"

// ---------------------------------------------------------------------------
// Stack entry and stack struct
// ---------------------------------------------------------------------------
/// Priority and material index share one word: the priority in the top 8 bits,
/// the material index in the bottom 24.
///
/// Neither needs more. The priority says which surface wins where two
/// dielectrics overlap and the loader authors it as 0 or 10; a scene with more
/// than sixteen million materials has run out of other things first. Packing
/// them takes the entry from 12 bytes to 8 and the stack from 52 to 36, and this
/// stack is carried per path on both backends -- inside OptiX's PerRayData,
/// where every byte is a byte of continuation stack, and in Metal's per-pixel
/// side table. See docs/open-perf.md for what a byte of PerRayData costs.
enum : unsigned int
{
    IOR_ENTRY_MATERIAL_BITS = 24u,
    IOR_ENTRY_MATERIAL_MASK = (1u << IOR_ENTRY_MATERIAL_BITS) - 1u
};

struct IorStackEntry
{
    float ior;
    // Which material this medium came from, and at what priority. Absorption is
    // a property of the volume, so a path inside one has to be able to look its
    // coefficients back up; carrying the index is cheaper than carrying a float3
    // sigma_t.
    unsigned int packed;
};

DEVICE_FUNC unsigned int ior_entry_priority(const THREAD_REF IorStackEntry& e)
{
    return e.packed >> IOR_ENTRY_MATERIAL_BITS;
}

DEVICE_FUNC unsigned int ior_entry_material(const THREAD_REF IorStackEntry& e)
{
    return e.packed & IOR_ENTRY_MATERIAL_MASK;
}

/// Both fields are masked rather than asserted: this runs on three compilers,
/// two of which have no way to complain, and a priority that overflowed into the
/// material index would be a wrong medium rather than a loud failure.
DEVICE_FUNC unsigned int ior_entry_pack(unsigned int priority, unsigned int material_index)
{
    return ((priority & 0xFFu) << IOR_ENTRY_MATERIAL_BITS) | (material_index & IOR_ENTRY_MATERIAL_MASK);
}

enum : int
{
    IOR_STACK_SIZE = 4
};

struct IorStack
{
    IorStackEntry entries[IOR_STACK_SIZE]; // 32 bytes
    int top;                               //  4 bytes (-1 = empty = air)
};

// ---------------------------------------------------------------------------
// ior_stack_init -- Reset the stack to empty (air)
// ---------------------------------------------------------------------------
DEVICE_FUNC void ior_stack_init(THREAD_REF IorStack& stack)
{
    stack.top = -1;
}

// ---------------------------------------------------------------------------
// ior_stack_current_ior -- Return the IOR of the medium the ray is in
// ---------------------------------------------------------------------------
DEVICE_FUNC float ior_stack_current_ior(const THREAD_REF IorStack& stack)
{
    return (stack.top >= 0) ? stack.entries[stack.top].ior : 1.0f;
}

// ---------------------------------------------------------------------------
// ior_stack_push -- Push a new medium onto the stack (entering geometry)
// ---------------------------------------------------------------------------
DEVICE_FUNC void ior_stack_push(THREAD_REF IorStack& stack,
                                 unsigned int priority, float ior,
                                 unsigned int material_index)
{
    if (stack.top < IOR_STACK_SIZE - 1)
    {
        stack.top++;
        stack.entries[stack.top].ior = ior;
        stack.entries[stack.top].packed = ior_entry_pack(priority, material_index);
    }
}

// Which material's volume the ray is currently inside, or 0xFFFFFFFF for air.
DEVICE_FUNC unsigned int ior_stack_current_material(const THREAD_REF IorStack& stack)
{
    return (stack.top >= 0) ? ior_entry_material(stack.entries[stack.top]) : 0xFFFFFFFFu;
}

// ---------------------------------------------------------------------------
// The two ways this stack loses a path, asked as questions.
//
// Both failures are silent by construction: push does nothing when it is full,
// and pop returns success having found nothing. Either one leaves the path
// carrying the wrong medium -- or none -- for the rest of its life, and the
// absorption it applies afterwards belongs to something else. The bathroom did
// exactly that for as long as it existed and nothing said so.
//
// Predicates rather than return values because the two callers are in two
// different renderers and a changed signature is a changed OptiX payload; asking
// first costs a loop over at most four entries, on a path that is already inside
// "this bounce was a transmission through a solid".
// ---------------------------------------------------------------------------
DEVICE_FUNC bool ior_stack_full(const THREAD_REF IorStack& stack)
{
    return stack.top >= IOR_STACK_SIZE - 1;
}

/// Whether ior_stack_pop would find anything to remove. False means the path is
/// leaving something it never entered.
DEVICE_FUNC bool ior_stack_can_pop(const THREAD_REF IorStack& stack,
                                   unsigned int priority,
                                   unsigned int material_index)
{
    for (int i = stack.top; i >= 0; i--)
    {
        if (ior_entry_material(stack.entries[i]) == (material_index & IOR_ENTRY_MATERIAL_MASK) ||
            ior_entry_priority(stack.entries[i]) == (priority & 0xFFu))
        {
            return true;
        }
    }
    return false;
}

// ---------------------------------------------------------------------------
// ior_stack_pop -- Remove the entry for the surface being left (exiting)
//
// Matched on the material first, and only then on the priority.
//
// Priority alone cannot identify the entry: it says which surface wins where two
// dielectrics overlap, not which object this is, and glTF gives no way to author
// it -- the loader assigns one value to everything transmissive. The bathroom
// has twelve such materials all at 10, so leaving the shower glass popped
// whichever of them happened to be on top: the bath water, a bubble, the lotion
// in the bottle. The path then carried the wrong medium, or none, for the rest
// of its life, and the absorption it applied afterwards belonged to something
// else.
//
// The priority search stays as the fallback for the case it was written for: a
// material that legitimately shares a priority with the one being left, where
// removing any of them leaves the same IOR on top.
//
// Searches from top to bottom. Shifts remaining entries down to fill the gap.
// Returns the new current IOR after popping.
// ---------------------------------------------------------------------------
DEVICE_FUNC float ior_stack_pop(THREAD_REF IorStack& stack,
                                 unsigned int priority,
                                 unsigned int material_index)
{
    // The surface this exit belongs to, if the path ever entered it.
    for (int i = stack.top; i >= 0; i--)
    {
        if (ior_entry_material(stack.entries[i]) == (material_index & IOR_ENTRY_MATERIAL_MASK))
        {
            for (int j = i; j < stack.top; j++)
            {
                stack.entries[j] = stack.entries[j + 1];
            }
            stack.top--;
            return ior_stack_current_ior(stack);
        }
    }
    // Find the entry with matching priority (search from top)
    for (int i = stack.top; i >= 0; i--)
    {
        if (ior_entry_priority(stack.entries[i]) == (priority & 0xFFu))
        {
            // Shift entries above the removed one down
            for (int j = i; j < stack.top; j++)
            {
                stack.entries[j] = stack.entries[j + 1];
            }
            stack.top--;
            return ior_stack_current_ior(stack);
        }
    }
    // Priority not found -- return current IOR unchanged
    return ior_stack_current_ior(stack);
}

// ---------------------------------------------------------------------------
// ior_stack_peek_after_pop -- Return what current_ior would be after popping
//                             the given priority, WITHOUT modifying the stack.
//
// Used to compute exterior_ior when exiting a medium: the exterior is the
// medium we'll be in *after* leaving, which is the stack without this entry.
// ---------------------------------------------------------------------------
DEVICE_FUNC float ior_stack_peek_after_pop(const THREAD_REF IorStack& stack,
                                            unsigned int priority)
{
    // Find the entry with matching priority
    int found = -1;
    for (int i = stack.top; i >= 0; i--)
    {
        if (ior_entry_priority(stack.entries[i]) == (priority & 0xFFu))
        {
            found = i;
            break;
        }
    }

    if (found < 0)
    {
        // Not found -- current IOR unchanged
        return ior_stack_current_ior(stack);
    }

    // The new top after removing 'found' would be:
    // If found == top, new top is top-1
    // If found < top, top stays the same (entry above shifts down)
    if (found == stack.top)
    {
        return (stack.top - 1 >= 0) ? stack.entries[stack.top - 1].ior : 1.0f;
    }
    else
    {
        // Removing a non-top entry doesn't change the top-of-stack IOR
        return stack.entries[stack.top].ior;
    }
}

#endif // STRELKA_IOR_STACK_H
