#pragma once

#include "common.h"
#include "computepass.h"
#include "debugviewparam.h"

#include <vector>

namespace oka
{
struct DebugDesc
{
    Debugviewparam constants;
    Image* normal = VK_NULL_HANDLE;
    Image* motion = VK_NULL_HANDLE;
    Image* debug = VK_NULL_HANDLE;
    Image* pathTracer = VK_NULL_HANDLE;

    Image* input = VK_NULL_HANDLE;
    Image* output = VK_NULL_HANDLE;
};

using DebugViewBase = ComputePass<Debugviewparam>;
class DebugView : public DebugViewBase
{
public:
    DebugView(const SharedContext& ctx);
    ~DebugView();
    void initialize();
    void execute(VkCommandBuffer& cmd, const DebugDesc& desc, uint32_t width, uint32_t height, uint64_t frameIndex);
};
} // namespace oka
