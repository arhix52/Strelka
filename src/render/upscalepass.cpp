#include "upscalepass.h"

namespace oka
{
UpscalePass::UpscalePass(const SharedContext& ctx) : UpscalePassBase(ctx)
{
}
UpscalePass::~UpscalePass()
{
    vkDestroySampler(mSharedCtx.mDevice, mUpscaleSampler, nullptr);
}
void UpscalePass::initialize()
{
    // sampler
    VkSamplerCreateInfo samplerInfo{};
    samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerInfo.magFilter = VK_FILTER_LINEAR;
    samplerInfo.minFilter = VK_FILTER_LINEAR;
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.anisotropyEnable = VK_TRUE;
    samplerInfo.maxAnisotropy = 1.0f;
    samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
    samplerInfo.unnormalizedCoordinates = VK_FALSE;
    samplerInfo.compareEnable = VK_FALSE;
    samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;
    samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;

    VkResult res = vkCreateSampler(mSharedCtx.mDevice, &samplerInfo, nullptr, &mUpscaleSampler);
    if (res != VK_SUCCESS)
    {
        // error
        assert(0);
    }

    UpscalePassBase::initialize("shaders/upscalepass.hlsl");
}
void UpscalePass::execute(VkCommandBuffer& cmd, const UpscaleDesc& desc, uint32_t width, uint32_t height, uint64_t frameIndex)
{
    auto& param = mShaderParamFactory.getNextShaderParameters(frameIndex);
    {
        param.setConstants(desc.constants);
        param.setTexture("input", mSharedCtx.mResManager->getView(desc.input));
        param.setTexture("output", mSharedCtx.mResManager->getView(desc.output));
        param.setSampler("upscaleSampler", mUpscaleSampler);
    }
    int frameVersion = frameIndex % MAX_FRAMES_IN_FLIGHT;
    Result res = updatePipeline(frameVersion);
    assert(res == Result::eOk);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, getPipeline(frameVersion));
    VkDescriptorSet descSet = param.getDescriptorSet(frameIndex);
    vkCmdBindDescriptorSets(
        cmd, VK_PIPELINE_BIND_POINT_COMPUTE, getPipeLineLayout(frameVersion), 0, 1, &descSet, 0, nullptr);
    const uint32_t dispX = (width + 15) / 16;
    const uint32_t dispY = (height + 15) / 16;
    vkCmdDispatch(cmd, dispX, dispY, 1);
}

} // namespace oka
