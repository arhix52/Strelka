#pragma once

#include <atomic>
#include <cstdint>

namespace oka
{

struct LoadProgress
{
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
