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
struct IorStackEntry
{
    unsigned int priority;
    float ior;
    // Which material this medium came from. Absorption is a property of the
    // volume, so a path inside one has to be able to look its coefficients back
    // up; carrying the index is cheaper than carrying a float3 sigma_t, and the
    // stack sits in the OptiX payload and in a per-pixel device buffer.
    unsigned int material_index;
};

#define IOR_STACK_SIZE 4

struct IorStack
{
    IorStackEntry entries[IOR_STACK_SIZE]; // 48 bytes
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
        stack.entries[stack.top].priority = priority;
        stack.entries[stack.top].ior = ior;
        stack.entries[stack.top].material_index = material_index;
    }
}

// Which material's volume the ray is currently inside, or 0xFFFFFFFF for air.
DEVICE_FUNC unsigned int ior_stack_current_material(const THREAD_REF IorStack& stack)
{
    return (stack.top >= 0) ? stack.entries[stack.top].material_index : 0xFFFFFFFFu;
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
        if (stack.entries[i].material_index == material_index)
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
        if (stack.entries[i].priority == priority)
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
        if (stack.entries[i].priority == priority)
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
