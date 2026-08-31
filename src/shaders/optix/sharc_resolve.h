#pragma once

#include <cuda_runtime.h>
#include <stdint.h>

struct SharcEntry;

/// Everything the resolve pass needs that is not the table itself.
///
/// A struct rather than a parameter list because it grew one: reprojection needs
/// both cameras, and a signature of nine scalars is a signature where two of
/// them get swapped.
struct SharcResolveParams
{
    uint32_t capacity = 0u;
    /// Frames the ordinary half of a voxel averages over.
    uint32_t accumFrameNumMax = 32u;
    /// Frames an entry survives with nothing deposited into it.
    uint32_t staleFrameNumMax = 64u;
    /// Accumulation window for the responsive part of the signal.
    uint32_t responsiveFrameNumMax = 4u;

    /// Where the eye is, and where it was when the previous frame resolved.
    /// Reprojection compares a voxel's distance to both to work out which way
    /// the grid level moved under it. See oka::sharc::adjacentLevelKey.
    float cameraPosition[3] = { 0.0f, 0.0f, 0.0f };
    float cameraPositionPrev[3] = { 0.0f, 0.0f, 0.0f };
    /// False on the first frame and whenever the eye has not moved, in which
    /// case there is nothing to reproject and the probe is skipped.
    bool reproject = false;
};

/// Between launches, fold the frame's deposits into resolved radiance and evict
/// stale slots.
extern "C" void sharcResolve(SharcEntry* entries, const SharcResolveParams& params, cudaStream_t stream);

/// Count entries in use, for the editor's occupancy readout.
///
/// Occupancy is the one number that says whether the cache is working: the SDK's
/// guidance is 10-20% with a static camera, and a table pinned near full is
/// thrashing -- inserting and evicting faster than entries ever resolve, which
/// costs the atomics and returns nothing. Asynchronous, into a device counter
/// the caller owns; nothing waits on it.
extern "C" void sharcCountOccupancy(
    const SharcEntry* entries, uint32_t capacity, uint32_t* deviceCounter, cudaStream_t stream);
