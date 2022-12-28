#pragma once
#include "common.h"

#include <array>
#include <unordered_map>
#include <vector>

namespace oka
{
template <typename T>
class ShaderParameters
{
protected:
    VkDevice mDevice;
    ResourceManager* mResManager = nullptr;
    std::array<T, MAX_FRAMES_IN_FLIGHT> mConstants = {};
    Buffer* mConstantBuffer = nullptr;
    uint32_t mCbBinding = 0;

    VkDescriptorSetLayout mDescriptorSetLayout = VK_NULL_HANDLE;
    std::array<VkDescriptorSet, MAX_FRAMES_IN_FLIGHT> mDescriptorSets;

    std::array<bool, MAX_FRAMES_IN_FLIGHT> needDesciptorSetUpdate = { false, false, false };
    std::array<bool, MAX_FRAMES_IN_FLIGHT> needConstantsUpdate = { false, false, false };

    std::vector<ShaderManager::ResourceDesc>& mResourcesDescs;
    std::unordered_map<std::string, ShaderManager::ResourceDesc>& mNameToDesc;

    struct ResourceDescriptor
    {
        ShaderManager::ResourceType type;
        union Handle
        {
            VkImageView imageView;
            VkBuffer buffer;
            VkSampler sampler;
        };
        bool isArray = false;
        std::vector<Handle> handles;
    };
    std::array<std::unordered_map<std::string, ResourceDescriptor>, MAX_FRAMES_IN_FLIGHT> mResUpdate = {};

    std::array<std::vector<VkWriteDescriptorSet>, MAX_FRAMES_IN_FLIGHT> mDescriptorWrites = {};

    const size_t minUniformBufferOffsetAlignment = 256; // 0x40

    size_t getConstantBufferOffset(const uint32_t index)
    {
        const size_t structSize = sizeof(T);
        return ((structSize + minUniformBufferOffsetAlignment - 1) / minUniformBufferOffsetAlignment) *
               minUniformBufferOffsetAlignment * index;
    }

    Result createConstantBuffers()
    {
        const size_t structSize = sizeof(T);
        // constan buffer size on must be a multiple of VkPhysicalDeviceLimits::minUniformBufferOffsetAlignment (256)
        const VkDeviceSize bufferSize =
            ((structSize + minUniformBufferOffsetAlignment - 1) / minUniformBufferOffsetAlignment) *
            minUniformBufferOffsetAlignment * MAX_FRAMES_IN_FLIGHT;
        mConstantBuffer =
            mResManager->createBuffer(bufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        if (!mConstantBuffer)
        {
            return Result::eOutOfMemory;
        }
        return Result::eOk;
    }

    void writeConstantBufferDescriptors()
    {
        assert(mConstantBuffer);
        std::vector<VkWriteDescriptorSet> descriptorWrites;
        std::vector<VkDescriptorBufferInfo> bufferInfos(MAX_FRAMES_IN_FLIGHT);
        for (uint32_t descIndex = 0; descIndex < MAX_FRAMES_IN_FLIGHT; ++descIndex)
        {
            VkDescriptorSet& dstDescSet = mDescriptorSets[descIndex];

            VkDescriptorBufferInfo& bufferInfo = bufferInfos[descIndex];
            bufferInfo.buffer = mResManager->getVkBuffer(mConstantBuffer);
            bufferInfo.offset = getConstantBufferOffset(descIndex); // we have one buffer, but split it per frame
            bufferInfo.range = sizeof(T);

            VkWriteDescriptorSet descWrite{};
            descWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descWrite.dstSet = dstDescSet;
            descWrite.dstBinding = mCbBinding;
            descWrite.dstArrayElement = 0;
            descWrite.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            descWrite.descriptorCount = 1; // support only one const buffer per shader param
            descWrite.pBufferInfo = &bufferInfo;

            descriptorWrites.push_back(descWrite);
        }

        vkUpdateDescriptorSets(
            mDevice, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
    }

    Result createDescriptorSets(const VkDescriptorPool& descriptorPool)
    {
        std::vector<VkDescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, mDescriptorSetLayout);
        VkDescriptorSetAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorPool = descriptorPool;
        allocInfo.descriptorSetCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);
        allocInfo.pSetLayouts = layouts.data();

        VkResult res = vkAllocateDescriptorSets(mDevice, &allocInfo, mDescriptorSets.data());
        if (res != VK_SUCCESS)
        {
            return Result::eFail;
        }
        return Result::eOk;
    }

    Result updateDescriptorSet(uint32_t descIndex)
    {
        VkDescriptorSet& dstDescSet = mDescriptorSets[descIndex];

        auto& resourcesToUpdate = mResUpdate[descIndex];

        uint32_t buffCount = 0;
        uint32_t texCount = 0;

        for (auto& currRes : resourcesToUpdate)
        {
            std::string name = currRes.first;
            auto it = mNameToDesc.find(name);
            if (it != mNameToDesc.end())
            {
                ShaderManager::ResourceDesc& resDesc = it->second;
                switch (currRes.second.type)
                {
                case ShaderManager::ResourceType::eConstantBuffer:
                case ShaderManager::ResourceType::eByteAddressBuffer:
                case ShaderManager::ResourceType::eStructuredBuffer: {
                    buffCount += currRes.second.isArray ? resDesc.arraySize : 1;
                    break;
                }
                case ShaderManager::ResourceType::eTexture3D:
                case ShaderManager::ResourceType::eRWTexture3D:
                case ShaderManager::ResourceType::eCubeMap:
                case ShaderManager::ResourceType::eRWCubeMap:
                case ShaderManager::ResourceType::eTexture2D:
                case ShaderManager::ResourceType::eRWTexture2D:
                case ShaderManager::ResourceType::eSampler: {
                    texCount += currRes.second.isArray ? resDesc.arraySize : 1;
                    break;
                }
                default:
                    break;
                }
            }
        }

        std::vector<VkDescriptorImageInfo> imageInfos(texCount);
        uint32_t imageInfosOffset = 0;
        std::vector<VkDescriptorBufferInfo> bufferInfos(buffCount);
        uint32_t bufferInfosOffset = 0;

        std::vector<VkWriteDescriptorSet> descriptorWrites;
        for (auto& currRes : resourcesToUpdate)
        {
            std::string name = currRes.first;
            ResourceDescriptor& descriptor = currRes.second;
            auto it = mNameToDesc.find(name);
            if (it != mNameToDesc.end())
            {
                ShaderManager::ResourceDesc& resDesc = it->second;
                const uint32_t descriptorCount = resDesc.isArray ? resDesc.arraySize : 1;

                switch (resDesc.type)
                {
                case ShaderManager::ResourceType::eTexture2D: {
                    for (uint32_t i = 0; i < descriptorCount; ++i)
                    {
                        imageInfos[imageInfosOffset + i].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                        imageInfos[imageInfosOffset + i].imageView =
                            (i < descriptor.handles.size()) ? descriptor.handles[i].imageView : VK_NULL_HANDLE;
                        imageInfos[imageInfosOffset + i].sampler = VK_NULL_HANDLE;
                    }

                    VkWriteDescriptorSet descWrite{};
                    descWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                    descWrite.dstSet = dstDescSet;
                    descWrite.dstBinding = resDesc.binding;
                    descWrite.dstArrayElement = 0;
                    descWrite.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
                    descWrite.descriptorCount = descriptorCount;
                    descWrite.pImageInfo = &imageInfos[imageInfosOffset];

                    imageInfosOffset += descriptorCount;

                    descriptorWrites.push_back(descWrite);
                    break;
                }
                case ShaderManager::ResourceType::eRWTexture2D: {
                    for (uint32_t i = 0; i < descriptorCount; ++i)
                    {
                        imageInfos[imageInfosOffset + i].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
                        imageInfos[imageInfosOffset + i].imageView = descriptor.handles[i].imageView;
                        imageInfos[imageInfosOffset + i].sampler = VK_NULL_HANDLE;
                    }

                    VkWriteDescriptorSet descWrite{};
                    descWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                    descWrite.dstSet = dstDescSet;
                    descWrite.dstBinding = resDesc.binding;
                    descWrite.dstArrayElement = 0;
                    descWrite.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
                    descWrite.descriptorCount = descriptorCount;
                    descWrite.pImageInfo = &imageInfos[imageInfosOffset];

                    imageInfosOffset += descriptorCount;

                    descriptorWrites.push_back(descWrite);
                    break;
                }
                case ShaderManager::ResourceType::eCubeMap:
                case ShaderManager::ResourceType::eTexture3D: {
                    for (uint32_t i = 0; i < descriptorCount; ++i)
                    {
                        imageInfos[imageInfosOffset + i].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                        imageInfos[imageInfosOffset + i].imageView =
                            (i < descriptor.handles.size()) ? descriptor.handles[i].imageView : VK_NULL_HANDLE;
                        imageInfos[imageInfosOffset + i].sampler = VK_NULL_HANDLE;
                    }

                    VkWriteDescriptorSet descWrite{};
                    descWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                    descWrite.dstSet = dstDescSet;
                    descWrite.dstBinding = resDesc.binding;
                    descWrite.dstArrayElement = 0;
                    descWrite.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
                    descWrite.descriptorCount = descriptorCount;
                    descWrite.pImageInfo = &imageInfos[imageInfosOffset];

                    imageInfosOffset += descriptorCount;

                    descriptorWrites.push_back(descWrite);
                    break;
                }
                case ShaderManager::ResourceType::eRWTexture3D: {
                    for (uint32_t i = 0; i < descriptorCount; ++i)
                    {
                        imageInfos[imageInfosOffset + i].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
                        imageInfos[imageInfosOffset + i].imageView = descriptor.handles[i].imageView;
                        imageInfos[imageInfosOffset + i].sampler = VK_NULL_HANDLE;
                    }

                    VkWriteDescriptorSet descWrite{};
                    descWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                    descWrite.dstSet = dstDescSet;
                    descWrite.dstBinding = resDesc.binding;
                    descWrite.dstArrayElement = 0;
                    descWrite.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
                    descWrite.descriptorCount = descriptorCount;
                    descWrite.pImageInfo = &imageInfos[imageInfosOffset];

                    imageInfosOffset += descriptorCount;

                    descriptorWrites.push_back(descWrite);
                    break;
                }
                case ShaderManager::ResourceType::eSampler: {
                    // for (uint32_t i = 0; i < descriptorCount; ++i)
                    //{
                    //    imageInfos[imageInfosOffset + i].sampler = (i < descriptor.handles.size()) ?
                    //    descriptor.handles[i].sampler : VK_NULL_HANDLE;
                    //}
                    assert(descriptor.handles.size() > 0);
                    for (uint32_t i = 0; i < (uint32_t)descriptor.handles.size(); ++i)
                    {
                        imageInfos[imageInfosOffset + i].imageView = VK_NULL_HANDLE;
                        imageInfos[imageInfosOffset + i].sampler = descriptor.handles[i].sampler;
                    }
                    VkWriteDescriptorSet descWrite{};
                    descWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                    descWrite.dstSet = dstDescSet;
                    descWrite.dstBinding = resDesc.binding;
                    descWrite.dstArrayElement = 0;
                    descWrite.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
                    // descWrite.descriptorCount = descriptorCount;
                    descWrite.descriptorCount = (uint32_t)descriptor.handles.size();
                    descWrite.pImageInfo = &imageInfos[imageInfosOffset];

                    imageInfosOffset += descriptorCount;

                    descriptorWrites.push_back(descWrite);
                    break;
                }
                case ShaderManager::ResourceType::eByteAddressBuffer:
                case ShaderManager::ResourceType::eRWStructuredBuffer:
                case ShaderManager::ResourceType::eStructuredBuffer: {
                    for (uint32_t i = 0; i < descriptorCount; ++i)
                    {
                        bufferInfos[bufferInfosOffset + i].buffer = descriptor.handles[i].buffer;
                        bufferInfos[bufferInfosOffset + i].offset = 0;
                        bufferInfos[bufferInfosOffset + i].range = VK_WHOLE_SIZE;
                    }

                    VkWriteDescriptorSet descWrite{};
                    descWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                    descWrite.dstSet = dstDescSet;
                    descWrite.dstBinding = resDesc.binding;
                    descWrite.dstArrayElement = 0;
                    descWrite.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                    descWrite.descriptorCount = descriptorCount;
                    descWrite.pBufferInfo = &bufferInfos[bufferInfosOffset];

                    bufferInfosOffset += descriptorCount;

                    descriptorWrites.push_back(descWrite);
                    break;
                }
                default:
                    break;
                }
            }
            else
            {
                // not found
                return Result::eFail;
            }
        }

        vkUpdateDescriptorSets(
            mDevice, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
        needDesciptorSetUpdate[descIndex] = false;

        mResUpdate[descIndex].clear();

        return Result::eOk;
    }

public:
    ShaderParameters(VkDescriptorSetLayout descriptorSetLayout,
                     std::vector<ShaderManager::ResourceDesc>& resourcesDescs,
                     std::unordered_map<std::string, ShaderManager::ResourceDesc>& nameToDesc,
                     int cbBinding)
        : mCbBinding(cbBinding),
          mDescriptorSetLayout(descriptorSetLayout),
          mResourcesDescs(resourcesDescs),
          mNameToDesc(nameToDesc)
    {
    }

    virtual ~ShaderParameters()
    {
        assert(mResManager);
        if (mConstantBuffer)
        {
            mResManager->destroyBuffer(mConstantBuffer);
        }
    }

    VkDescriptorSet getDescriptorSet(const uint64_t frameIndex)
    {
        const uint32_t index = frameIndex % MAX_FRAMES_IN_FLIGHT;
        if (needDesciptorSetUpdate[index])
        {
            Result res = updateDescriptorSet(index);
            assert(res == Result::eOk);
            if (res != Result::eOk)
            {
                // TODO: report error
                printf("Error!\n");
            }
            needDesciptorSetUpdate[index] = false;
        }
        if (needConstantsUpdate[index])
        {
            void* data = mResManager->getMappedMemory(mConstantBuffer);
            assert(data);
            // offset to current frame
            memcpy((void*)((char*)data + getConstantBufferOffset(index)), &mConstants[index], sizeof(T));
            needConstantsUpdate[index] = false;
        }
        return mDescriptorSets[index];
    }

    Result create(const SharedContext& ctx)
    {
        mDevice = ctx.mDevice;
        mResManager = ctx.mResManager;

        Result res = createDescriptorSets(ctx.mDescriptorPool);

        res = createConstantBuffers();
        // Now we support only 1 constant buffer per shader
        writeConstantBufferDescriptors();

        return res;
    }

    // setters
    void setConstants(const T& constants)
    {
        for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
        {
            mConstants[i] = constants;
            needConstantsUpdate[i] = true;
        }
    }

    void setBuffer(const std::string& name, VkBuffer buff)
    {
        for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
        {
            ResourceDescriptor resDescriptor{};
            resDescriptor.type = ShaderManager::ResourceType::eStructuredBuffer;
            resDescriptor.isArray = false;
            resDescriptor.handles.resize(1);
            resDescriptor.handles[0].buffer = buff;
            mResUpdate[i][name] = resDescriptor;
            needDesciptorSetUpdate[i] = true;
        }
    }

    void setTexture(const std::string& name, VkImageView view)
    {
        for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
        {
            ResourceDescriptor resDescriptor{};
            resDescriptor.type = ShaderManager::ResourceType::eTexture2D;
            resDescriptor.isArray = false;
            resDescriptor.handles.resize(1);
            resDescriptor.handles[0].imageView = view;
            mResUpdate[i][name] = resDescriptor;
            needDesciptorSetUpdate[i] = true;
        }
    }

    void setTextures(const std::string& name, std::vector<Image*>& images)
    {
        for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
        {
            ResourceDescriptor resDescriptor{};
            resDescriptor.type = ShaderManager::ResourceType::eTexture2D;
            resDescriptor.isArray = true;
            resDescriptor.handles.resize(images.size());
            for (uint32_t j = 0; j < images.size(); ++j)
            {
                resDescriptor.handles[j].imageView = mResManager->getView(images[j]);
            }
            mResUpdate[i][name] = resDescriptor;
            needDesciptorSetUpdate[i] = true;
        }
    }

    void setTextures3d(const std::string& name, std::vector<Image*>& images)
    {
        for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
        {
            ResourceDescriptor resDescriptor{};
            resDescriptor.type = ShaderManager::ResourceType::eTexture3D;
            resDescriptor.isArray = true;
            resDescriptor.handles.resize(images.size());
            for (uint32_t j = 0; j < images.size(); ++j)
            {
                resDescriptor.handles[j].imageView = mResManager->getView(images[j]);
            }
            mResUpdate[i][name] = resDescriptor;
            needDesciptorSetUpdate[i] = true;
        }
    }

    void setCubeMap(const std::string& name, VkImageView view)
    {
        for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
        {
            ResourceDescriptor resDescriptor{};
            resDescriptor.type = ShaderManager::ResourceType::eCubeMap;
            resDescriptor.isArray = false;
            resDescriptor.handles.resize(1);
            resDescriptor.handles[0].imageView = view;
            mResUpdate[i][name] = resDescriptor;
            needDesciptorSetUpdate[i] = true;
        }
    }

    void setSampler(const std::string& name, VkSampler sampler)
    {
        for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
        {
            ResourceDescriptor resDescriptor{};
            resDescriptor.type = ShaderManager::ResourceType::eSampler;
            resDescriptor.isArray = false;
            resDescriptor.handles.resize(1);
            resDescriptor.handles[0].sampler = sampler;
            mResUpdate[i][name] = resDescriptor;
            needDesciptorSetUpdate[i] = true;
        }
    }

    void setSamplers(const std::string& name, std::vector<VkSampler>& samplers)
    {
        for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
        {
            ResourceDescriptor resDescriptor{};
            resDescriptor.type = ShaderManager::ResourceType::eSampler;
            resDescriptor.isArray = true;
            resDescriptor.handles.resize(samplers.size());
            for (uint32_t j = 0; j < samplers.size(); ++j)
            {
                resDescriptor.handles[j].sampler = samplers[j];
            }
            mResUpdate[i][name] = resDescriptor;
            needDesciptorSetUpdate[i] = true;
        }
    }
};

template <typename T, int NumFrames = MAX_FRAMES_IN_FLIGHT>
class ShaderParametersFactory
{
    std::vector<ShaderParameters<T>*> mShaderParams;
    uint64_t mCurrentFrameIndex = 0;
    uint64_t mIntraFrameIndex = 0;

    SharedContext mSharedCtx;
    int mShaderId = -1;
    int mCbBinding = -1;
    VkDescriptorSetLayout mDescriptorSetLayout = VK_NULL_HANDLE;

    std::vector<ShaderManager::ResourceDesc> mResourcesDescs;
    std::unordered_map<std::string, ShaderManager::ResourceDesc> mNameToDesc;

    Result createDescriptorSetLayout()
    {
        std::vector<VkDescriptorSetLayoutBinding> bindings;
        bindings.reserve(mResourcesDescs.size());
        uint32_t bindingIdx = 0;
        for (const auto& desc : mResourcesDescs)
        {
            VkDescriptorType descType = VK_DESCRIPTOR_TYPE_MAX_ENUM;
            switch (desc.type)
            {
            case ShaderManager::ResourceType::eConstantBuffer: {
                descType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
                mCbBinding = bindingIdx;
                break;
            }
            case ShaderManager::ResourceType::eTexture2D: {
                descType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
                break;
            }
            case ShaderManager::ResourceType::eRWTexture2D: {
                descType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
                break;
            }
            case ShaderManager::ResourceType::eStructuredBuffer:
            case ShaderManager::ResourceType::eRWStructuredBuffer: {
                descType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                break;
            }
            case ShaderManager::ResourceType::eByteAddressBuffer: {
                descType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                break;
            }
            case ShaderManager::ResourceType::eSampler: {
                descType = VK_DESCRIPTOR_TYPE_SAMPLER;
                break;
            }
            case ShaderManager::ResourceType::eCubeMap:
            case ShaderManager::ResourceType::eTexture3D: {
                descType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
                break;
            }
            case ShaderManager::ResourceType::eRWCubeMap:
            case ShaderManager::ResourceType::eRWTexture3D: {
                descType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
                break;
            }
            default:
                assert(0);
                break;
            }
            VkDescriptorSetLayoutBinding layoutBinding{};
            layoutBinding.binding = desc.binding;
            layoutBinding.descriptorCount = desc.isArray ? desc.arraySize : 1;
            layoutBinding.descriptorType = descType;
            layoutBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT; // TODO: fix for graphics
            bindings.push_back(layoutBinding);
            ++bindingIdx;
        }

        VkDescriptorSetLayoutCreateInfo layoutInfo{};
        layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
        layoutInfo.pBindings = bindings.data();

        VkResult res = vkCreateDescriptorSetLayout(mSharedCtx.mDevice, &layoutInfo, nullptr, &mDescriptorSetLayout);
        if (res != VK_SUCCESS)
        {
            return Result::eFail;
        }
        return Result::eOk;
    }

public:
    ShaderParametersFactory(const SharedContext& ctx) : mSharedCtx(ctx)
    {
    }
    virtual ~ShaderParametersFactory()
    {
        vkDestroyDescriptorSetLayout(mSharedCtx.mDevice, mDescriptorSetLayout, nullptr);
    }

    VkDescriptorSetLayout getDescriptorSetLayout()
    {
        return mDescriptorSetLayout;
    }

    bool initialize(uint32_t shaderId)
    {
        assert(shaderId != (uint32_t)-1);
        mShaderId = shaderId;

        mNameToDesc.clear(); // remove previous values
        mResourcesDescs = mSharedCtx.mShaderManager->getResourcesDesc(shaderId);
        for (auto& desc : mResourcesDescs)
        {
            mNameToDesc[desc.name] = desc;
        }

        if (mDescriptorSetLayout)
        {
            vkDestroyDescriptorSetLayout(mSharedCtx.mDevice, mDescriptorSetLayout, nullptr);
        }

        Result res = createDescriptorSetLayout();
        assert(mCbBinding != -1);
        return res == Result::eOk;
    }

    ShaderParameters<T>& getNextShaderParameters(uint64_t frameIndex)
    {
        if (mCurrentFrameIndex < frameIndex)
        {
            // reset
            mCurrentFrameIndex = frameIndex;
            mIntraFrameIndex = 0;
        }

        if (mIntraFrameIndex >= mShaderParams.size())
        {
            auto param = new ShaderParameters<T>(mDescriptorSetLayout, mResourcesDescs, mNameToDesc, mCbBinding);
            Result res = param->create(mSharedCtx);
            assert(res == Result::eOk);
            mShaderParams.push_back(param);
        }
        ++mIntraFrameIndex;

        return *(mShaderParams[mIntraFrameIndex - 1]);
    }
};

} // namespace oka
