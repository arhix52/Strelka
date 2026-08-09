#pragma once

#include <atomic>
#include <cstdint>

namespace oka
{

/// Progress of a scene load, written by whoever is doing the work and read by
/// the UI once a frame.
///
/// A load crosses two threads and does so in two different ways: the glTF parse
/// runs on a worker, and the GPU-side build runs a chunk at a time on the main
/// loop. Both report here, which is why the fields are atomic even though only
/// one of the producers is actually concurrent -- one shared shape is easier to
/// keep honest than two that agree by convention.
///
/// Nothing but scalars crosses the boundary. The stage is an index into a table
/// the UI owns rather than a string, so there is nothing to allocate on a worker,
/// nothing to lock while reading, and nothing whose lifetime has to outlive the
/// load. A reader that catches `done` and `total` mid-update draws one frame with
/// a slightly wrong bar, which is the whole cost of not having a mutex here.
struct LoadProgress
{
    /// Listed in the order they actually run, because the UI turns this into a
    /// fraction by summing the weights of the stages before the current one. A
    /// declaration order that disagreed with the run order would make the bar
    /// jump backwards, which is exactly the kind of thing nobody would think to
    /// look for here.
    enum class Stage : uint32_t
    {
        Idle = 0,
        Reading,     ///< tinygltf pulling the file in -- one indivisible call
        Parsing,     ///< materials, cameras, the node graph
        Geometry,    ///< vertex and index buffers
        Textures,    ///< decode and upload
        Structures,  ///< BLAS and TLAS
        Environment, ///< env map, alias table, remaining setup
        Done,
        Count
    };

    std::atomic<uint32_t> stage{ (uint32_t)Stage::Idle };
    std::atomic<uint32_t> done{ 0 };
    /// Zero means the stage cannot say how much work it holds, and the UI should
    /// show it as indeterminate rather than as complete.
    std::atomic<uint32_t> total{ 0 };

    /// Set by the UI, read by both producers at chunk boundaries. A load that is
    /// abandoned has to stop rather than spend seconds finishing work into an
    /// object that will be thrown away.
    std::atomic<bool> cancelled{ false };

    void beginStage(Stage s, uint32_t itemCount = 0)
    {
        done.store(0, std::memory_order_relaxed);
        total.store(itemCount, std::memory_order_relaxed);
        stage.store((uint32_t)s, std::memory_order_release);
    }

    void step(uint32_t n = 1)
    {
        done.fetch_add(n, std::memory_order_relaxed);
    }

    Stage currentStage() const
    {
        return (Stage)stage.load(std::memory_order_acquire);
    }

    bool isCancelled() const
    {
        return cancelled.load(std::memory_order_relaxed);
    }

    void cancel()
    {
        cancelled.store(true, std::memory_order_relaxed);
    }

    void reset()
    {
        cancelled.store(false, std::memory_order_relaxed);
        beginStage(Stage::Idle, 0);
    }
};

} // namespace oka
