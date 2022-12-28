#pragma once
#include "debugUtils.h"
#include "gbuffer.h"

#include <scene/scene.h>
#include <vulkan/vulkan.h>

#include <array>
#include <resourcemanager.h>
#include <vector>

namespace nevk
{
class GbufferPass
{
private:
    struct UniformBufferObject
    {
        alignas(16) glm::mat4 viewToProj;
        alignas(16) glm::mat4 worldToView;
        alignas(16) glm::mat4 prevViewToProj;
        alignas(16) glm::mat4 prevWorldToView;
        alignas(16) glm::float3 CameraPos;
        float pad;
    };

    struct InstancePushConstants
    {
        int32_t instanceId = -1;
    };

    static constexpr int MAX_FRAMES_IN_FLIGHT = 3;

    VkDevice mDevice;
    VkPipeline mPipeline;
    VkPipelineLayout mPipelineLayout;
    VkRenderPass mRenderPass;
    VkDescriptorSetLayout mDescriptorSetLayout;

    bool mEnableValidation = false;

    void beginLabel(VkCommandBuffer cmdBuffer, const char* labelName, const glm::float4& color)
    {
        if (mEnableValidation)
        {
            nevk::debug::beginLabel(cmdBuffer, labelName, color);
        }
    }

    void endLabel(VkCommandBuffer cmdBuffer)
    {
        if (mEnableValidation)
        {
            nevk::debug::endLabel(cmdBuffer);
        }
    }

    void updateDescriptorSets(uint32_t descSetIndex);

    VkShaderModule mVS, mPS;

    VkFormat mDepthFormat;

    ResourceManager* mResManager;
    VkDescriptorPool mDescriptorPool;
    std::vector<Buffer*> uniformBuffers;

    std::vector<VkSampler> mTextureSamplers;
    VkSampler mShadowSampler = VK_NULL_HANDLE;

    void createRenderPass();
    void createDescriptorSetLayout();
    void createDescriptorSets(VkDescriptorPool& descriptorPool);

    void createConstantBuffers();

    std::vector<VkDescriptorSet> mDescriptorSets;

    VkFramebuffer mFrameBuffers[MAX_FRAMES_IN_FLIGHT];

    static VkVertexInputBindingDescription getBindingDescription()
    {
        VkVertexInputBindingDescription bindingDescription{};
        bindingDescription.binding = 0;
        bindingDescription.stride = sizeof(Scene::Vertex);
        bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

        return bindingDescription;
    }

    static std::vector<VkVertexInputAttributeDescription> getAttributeDescriptions()
    {
        std::vector<VkVertexInputAttributeDescription> attributeDescriptions = {};

        VkVertexInputAttributeDescription attributeDescription = {};

        attributeDescription.binding = 0;
        attributeDescription.location = 0;
        attributeDescription.format = VK_FORMAT_R32G32B32_SFLOAT;
        attributeDescription.offset = offsetof(Scene::Vertex, pos);
        attributeDescriptions.emplace_back(attributeDescription);

        attributeDescription.binding = 0;
        attributeDescription.location = 1;
        attributeDescription.format = VK_FORMAT_R32_UINT;
        attributeDescription.offset = offsetof(Scene::Vertex, tangent);
        attributeDescriptions.emplace_back(attributeDescription);

        attributeDescription.binding = 0;
        attributeDescription.location = 2;
        attributeDescription.format = VK_FORMAT_R32_UINT;
        attributeDescription.offset = offsetof(Scene::Vertex, normal);
        attributeDescriptions.emplace_back(attributeDescription);

        attributeDescription.binding = 0;
        attributeDescription.location = 3;
        attributeDescription.format = VK_FORMAT_R32_UINT;
        attributeDescription.offset = offsetof(Scene::Vertex, uv);
        attributeDescriptions.emplace_back(attributeDescription);

        return attributeDescriptions;
    }

    VkShaderModule createShaderModule(const char* code, uint32_t codeSize);

public:
    int imageViewCounter = 0;

    std::vector<VkImageView> mTextureImageView;
    VkImageView mShadowImageView = VK_NULL_HANDLE;
    VkBuffer mMaterialBuffer = VK_NULL_HANDLE;
    VkBuffer mInstanceBuffer = VK_NULL_HANDLE;

    bool needDesciptorSetUpdate[MAX_FRAMES_IN_FLIGHT] = {false, false, false};

    VkPipelineLayout createGraphicsPipelineLayout();

    VkPipeline createGraphicsPipeline(VkShaderModule& vertShaderModule, VkShaderModule& fragShaderModule, VkPipelineLayout pipelineLayout, uint32_t width, uint32_t height);

    void createFrameBuffers(GBuffer& gbuffer, uint32_t index);

    void setShadowImageView(VkImageView shadowImageView);
    void setTextureImageView(const std::vector<VkImageView>& textureImageView);
    void setTextureSamplers(std::vector<VkSampler>& textureSamplers);
    void setShadowSampler(VkSampler shadowSampler);
    void setMaterialBuffer(VkBuffer materialBuffer);
    void setInstanceBuffer(VkBuffer instanceBuffer);

    void init(VkDevice& device, bool enableValidation, const char* vsCode, uint32_t vsCodeSize, const char* psCode, uint32_t psCodeSize, 
        VkDescriptorPool descpool, ResourceManager* resMngr, VkFormat depthFormat, uint32_t width, uint32_t height)
    {
        mDepthFormat = depthFormat;
        mEnableValidation = enableValidation;
        mDevice = device;
        mResManager = resMngr;
        mDescriptorPool = descpool;
        mVS = createShaderModule(vsCode, vsCodeSize);
        mPS = createShaderModule(psCode, psCodeSize);
        createConstantBuffers();

        createRenderPass();
        createDescriptorSetLayout();
        createDescriptorSets(mDescriptorPool);
        mPipelineLayout = createGraphicsPipelineLayout();
        mPipeline = createGraphicsPipeline(mVS, mPS, mPipelineLayout, width, height);
    }

    void onResize(GBuffer* gbuffer, uint32_t index);

    void onDestroy();

    void updateUniformBuffer(uint32_t currentImage, Scene& scene, uint32_t cameraIndex);

    GbufferPass(/* args */);
    ~GbufferPass();
    void record(VkCommandBuffer& cmd, VkBuffer vertexBuffer, VkBuffer indexBuffer, oka::Scene& scene, uint32_t width, uint32_t height, uint32_t imageIndex, uint32_t cameraIndex);
};
} // namespace oka
