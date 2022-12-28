#pragma once

#include "common.h"
#include "computepass.h"
#include "pathtracerparam.h"

namespace oka
{

struct ReductionDesc
{
    PathTracerParam constants;
    Buffer* sampleBuffer;
    // Buffer* compositingBuffer;
    Image* result;
};

using ReductionPassBase = ComputePass<PathTracerParam>;
class ReductionPass : public ReductionPassBase
{
public:
    ReductionPass(const SharedContext& ctx);
    ~ReductionPass();
    void initialize();
    void execute(VkCommandBuffer& cmd, const ReductionDesc& desc, uint32_t width, uint32_t height, uint64_t frameIndex);
};
} // namespace oka
