#pragma once

#include "common.h"
#include "computepass.h"
#include "tonemapparam.h"

#include <vector>

namespace oka
{
struct TonemapDesc
{
    Tonemapparam constants;
    Image* input;
    Image* output;
};

using TonemapBase = ComputePass<Tonemapparam>;
class Tonemap : public TonemapBase
{
public:
    Tonemap(const SharedContext& ctx);
    ~Tonemap();
    void initialize();
    void execute(VkCommandBuffer& cmd, const TonemapDesc& desc, uint32_t width, uint32_t height, uint64_t frameIndex);
};
} // namespace oka
