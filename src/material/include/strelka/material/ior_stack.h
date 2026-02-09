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
};

#define IOR_STACK_SIZE 4

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
                                 unsigned int priority, float ior)
{
    if (stack.top < IOR_STACK_SIZE - 1)
    {
        stack.top++;
        stack.entries[stack.top].priority = priority;
        stack.entries[stack.top].ior = ior;
    }
}

// ---------------------------------------------------------------------------
// ior_stack_pop -- Remove the entry matching the given priority (exiting)
//
// Searches from top to bottom. Shifts remaining entries down to fill the gap.
// Returns the new current IOR after popping.
// ---------------------------------------------------------------------------
DEVICE_FUNC float ior_stack_pop(THREAD_REF IorStack& stack,
                                 unsigned int priority)
{
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
