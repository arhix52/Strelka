#pragma once

#include "accumulationparam.h"
#include "common.h"
#include "computepass.h"

namespace oka
{
struct AccumulationDesc
{
    AccumulationParam constants;
    Image* input = VK_NULL_HANDLE;
    Image* history = VK_NULL_HANDLE;
    Image* output = VK_NULL_HANDLE;
};

using AccumulationBase = ComputePass<AccumulationParam>;
class Accumulation : public AccumulationBase
{
public:
    Accumulation(const SharedContext& ctx);
    ~Accumulation();
    void initialize();

    void execute(VkCommandBuffer& cmd, const AccumulationDesc& desc, uint32_t width, uint32_t height, uint64_t frameIndex);
};
} // namespace oka
