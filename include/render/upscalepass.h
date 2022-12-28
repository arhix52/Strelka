#pragma once

#include "common.h"
#include "computepass.h"
#include "upscalepassparam.h"

#include <vector>

namespace oka
{
struct UpscaleDesc
{
    Upscalepassparam constants;
    Image* input;
    Image* output;
};

using UpscalePassBase = ComputePass<Upscalepassparam>;
class UpscalePass : public UpscalePassBase
{
private:
    VkSampler mUpscaleSampler = VK_NULL_HANDLE;

public:
    UpscalePass(const SharedContext& ctx);
    ~UpscalePass();
    void initialize();
    void execute(VkCommandBuffer& cmd, const UpscaleDesc& desc, uint32_t width, uint32_t height, uint64_t frameIndex);
};
} // namespace oka
