#pragma once

#include "common.h"
#include "shaderparameters.h"

namespace oka
{
template <typename T>
class ComputePass
{
protected:
    const SharedContext& mSharedCtx;

    VkPipeline mPipelines[MAX_FRAMES_IN_FLIGHT] = { VK_NULL_HANDLE };
    VkPipelineLayout mPipelineLayouts[MAX_FRAMES_IN_FLIGHT] = { VK_NULL_HANDLE };
    bool mNeedUpdatePipeline[MAX_FRAMES_IN_FLIGHT] = { true, true, true };

    VkShaderModule mCS = VK_NULL_HANDLE;
    uint32_t mCSid = (uint32_t)-1;

    ShaderParametersFactory<T> mShaderParamFactory;

    VkShaderModule createShaderModule(const char* code, uint32_t codeSize)
    {
        VkShaderModuleCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        createInfo.codeSize = codeSize;
        createInfo.pCode = (uint32_t*)code;

        VkShaderModule shaderModule;
        if (vkCreateShaderModule(mSharedCtx.mDevice, &createInfo, nullptr, &shaderModule) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to create shader module!");
        }

        return shaderModule;
    }
    Result createComputePipeline(VkShaderModule& shaderModule, int frameVersion)
    {
        assert((frameVersion >= 0) && (frameVersion < MAX_FRAMES_IN_FLIGHT));
        VkPipelineShaderStageCreateInfo shaderStageInfo{};
        shaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        shaderStageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        shaderStageInfo.module = shaderModule;
        shaderStageInfo.pName = "main";

        VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
        pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutInfo.setLayoutCount = 1;
        VkDescriptorSetLayout layout = mShaderParamFactory.getDescriptorSetLayout();
        pipelineLayoutInfo.pSetLayouts = &layout;

        if (vkCreatePipelineLayout(mSharedCtx.mDevice, &pipelineLayoutInfo, nullptr, &mPipelineLayouts[frameVersion]) !=
            VK_SUCCESS)
        {
            return Result::eFail;
        }

        VkComputePipelineCreateInfo pipelineInfo{};
        pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        pipelineInfo.stage = shaderStageInfo;
        pipelineInfo.layout = mPipelineLayouts[frameVersion];
        pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;

        if (vkCreateComputePipelines(
                mSharedCtx.mDevice, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &mPipelines[frameVersion]) != VK_SUCCESS)
        {
            return Result::eFail;
        }
        return Result::eOk;
    }

public:
    ComputePass(const SharedContext& ctx) : mSharedCtx(ctx), mShaderParamFactory(ctx)
    {
    }
    virtual ~ComputePass()
    {
        onDestroy();
    }

    void updateShader(const char* code)
    {
        const char* csShaderCode = nullptr;
        uint32_t csShaderCodeSize = 0;
        mCSid = mSharedCtx.mShaderManager->loadShaderFromString(code, "computeMain", oka::ShaderManager::Stage::eCompute);
        assert(mCSid != -1);
        mSharedCtx.mShaderManager->getShaderCode(mCSid, csShaderCode, csShaderCodeSize);
        if (mCS)
        {
            vkDestroyShaderModule(mSharedCtx.mDevice, mCS, nullptr);
        }
        mCS = createShaderModule(csShaderCode, csShaderCodeSize);

        // mark as dirty, we need update pipelines
        for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
        {
            mNeedUpdatePipeline[i] = true;
        }
    }

    void initializeFromCode(const char* code)
    {
        updateShader(code);

        mShaderParamFactory.initialize(mCSid);
    }

    void initialize(const char* shaderFile)
    {
        const char* csShaderCode = nullptr;
        uint32_t csShaderCodeSize = 0;
        uint32_t csId =
            mSharedCtx.mShaderManager->loadShader(shaderFile, "computeMain", oka::ShaderManager::Stage::eCompute);
        mSharedCtx.mShaderManager->getShaderCode(csId, csShaderCode, csShaderCodeSize);
        mCS = createShaderModule(csShaderCode, csShaderCodeSize);

        mShaderParamFactory.initialize(csId);
    }

    Result updatePipeline(int frameVersion)
    {
        assert((frameVersion >= 0) && (frameVersion < MAX_FRAMES_IN_FLIGHT));
        if (mNeedUpdatePipeline[frameVersion])
        {
            vkDestroyPipeline(mSharedCtx.mDevice, mPipelines[frameVersion], nullptr);
            vkDestroyPipelineLayout(mSharedCtx.mDevice, mPipelineLayouts[frameVersion], nullptr);
            Result res = createComputePipeline(mCS, frameVersion);
            mNeedUpdatePipeline[frameVersion] = false;
            return res;
        }
        return Result::eOk;
    }

    VkPipeline getPipeline(int frameVersion)
    {
        assert((frameVersion >= 0) && (frameVersion < MAX_FRAMES_IN_FLIGHT));
        return mPipelines[frameVersion];
    }

    VkPipelineLayout getPipeLineLayout(int frameIndex)
    {
        assert((frameIndex >= 0) && (frameIndex < MAX_FRAMES_IN_FLIGHT));
        return mPipelineLayouts[frameIndex];
    }

    // void execute(VkCommandBuffer& cmd, uint32_t width, uint32_t height, uint32_t imageIndex)

    void onDestroy()
    {
        for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
        {
            if (mPipelines[i])
            {
                vkDestroyPipeline(mSharedCtx.mDevice, mPipelines[i], nullptr);
            }
            if (mPipelineLayouts[i])
            {
                vkDestroyPipelineLayout(mSharedCtx.mDevice, mPipelineLayouts[i], nullptr);
            }
        }
        vkDestroyShaderModule(mSharedCtx.mDevice, mCS, nullptr);
    }
};
} // namespace oka
