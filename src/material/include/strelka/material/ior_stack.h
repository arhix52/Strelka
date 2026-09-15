#ifndef STRELKA_IOR_STACK_H
#define STRELKA_IOR_STACK_H

#include "material_math.h"

enum : unsigned int
{
    IOR_ENTRY_MATERIAL_BITS = 24u,
    IOR_ENTRY_MATERIAL_MASK = (1u << IOR_ENTRY_MATERIAL_BITS) - 1u
};

struct IorStackEntry
{
    float ior;
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
static_assert(sizeof(IorStack) == 36, "IorStack is a tightly packed per-path GPU side table");

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

/// Exact material operations for callers that carry a material index. Equal
/// priorities are common and must not make an unmatched exit remove a different
/// enclosing medium.
DEVICE_FUNC bool ior_stack_has_material(const THREAD_REF IorStack& stack, unsigned int material_index)
{
    for (int i = stack.top; i >= 0; i--)
    {
        if (ior_entry_material(stack.entries[i]) == (material_index & IOR_ENTRY_MATERIAL_MASK))
        {
            return true;
        }
    }
    return false;
}

DEVICE_FUNC float ior_stack_pop_material(THREAD_REF IorStack& stack, unsigned int material_index)
{
    for (int i = stack.top; i >= 0; i--)
    {
        if (ior_entry_material(stack.entries[i]) == (material_index & IOR_ENTRY_MATERIAL_MASK))
        {
            for (int j = i; j < stack.top; j++)
            {
                stack.entries[j] = stack.entries[j + 1];
            }
            stack.top--;
            break;
        }
    }
    return ior_stack_current_ior(stack);
}

DEVICE_FUNC float ior_stack_peek_after_pop_material(const THREAD_REF IorStack& stack, unsigned int material_index)
{
    for (int i = stack.top; i >= 0; i--)
    {
        if (ior_entry_material(stack.entries[i]) == (material_index & IOR_ENTRY_MATERIAL_MASK))
        {
            return (i == stack.top) ? ((i > 0) ? stack.entries[i - 1].ior : 1.0f) : stack.entries[stack.top].ior;
        }
    }
    return ior_stack_current_ior(stack);
}

#endif // STRELKA_IOR_STACK_H
