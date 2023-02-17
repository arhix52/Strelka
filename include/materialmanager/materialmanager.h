#pragma once

#include <memory>
#include <stdint.h>
#include <string>
#include <vector>

namespace oka
{

class MaterialManager
{
    class Context;
    std::unique_ptr<Context> mContext;

public:
    struct Module;
    struct MaterialInstance;
    struct CompiledMaterial;

    struct TargetCode;
    struct TextureDescription;

    bool addMdlSearchPath(const char* paths[], uint32_t numPaths);

    Module* createModule(const char* file);
    Module* createMtlxModule(const char* file);
    void destroyModule(Module* module);

    MaterialInstance* createMaterialInstance(Module* module, const char* materialName);
    void destroyMaterialInstance(MaterialInstance* material);

    struct Param
    {
        enum class Type : uint32_t
        {
            eFloat = 0,
            eInt,
            eBool,
            eFloat2,
            eFloat3,
            eFloat4,
            eTexture
        };
        Type type;
        std::string name;
        std::vector<uint8_t> value;
    };

    void dumpParams(const TargetCode* targetCode, CompiledMaterial* material);
    bool setParam(TargetCode* targetCode, uint32_t materialIdx, CompiledMaterial* material, const Param& param);

    TextureDescription* createTextureDescription(const char* name, const char* gamma);
    const char* getTextureDbName(TextureDescription* texDesc);

    CompiledMaterial* compileMaterial(MaterialInstance* matInstance);
    void destroyCompiledMaterial(CompiledMaterial* compMaterial);
    const char* getName(CompiledMaterial* compMaterial);

    TargetCode* generateTargetCode(CompiledMaterial** materials, const uint32_t numMaterials);
    const char* getShaderCode(const TargetCode* targetCode, uint32_t materialId);

    uint32_t getReadOnlyBlockSize(const TargetCode* targetCode);
    const uint8_t* getReadOnlyBlockData(const TargetCode* targetCode);

    uint32_t getArgBufferSize(const TargetCode* targetCode);
    const uint8_t* getArgBufferData(const TargetCode* targetCode);

    uint32_t getResourceInfoSize(const TargetCode* targetCode);
    const uint8_t* getResourceInfoData(const TargetCode* targetCode);
    int registerResource(TargetCode* targetCode, int index);

    uint32_t getMdlMaterialSize(const TargetCode* targetCode);
    const uint8_t* getMdlMaterialData(const TargetCode* targetCode);

    uint32_t getArgBlockOffset(const TargetCode* targetCode, uint32_t materialId);
    uint32_t getReadOnlyOffset(const TargetCode* targetCode, uint32_t materialId);

    uint32_t getTextureCount(const TargetCode* targetCode, uint32_t materialId);
    const char* getTextureName(const TargetCode* targetCode, uint32_t materialId, uint32_t index);
    const float* getTextureData(const TargetCode* targetCode, uint32_t materialId, uint32_t index);
    const char* getTextureType(const TargetCode* targetCode, uint32_t materialId, uint32_t index);
    uint32_t getTextureWidth(const TargetCode* targetCode, uint32_t materialId, uint32_t index);
    uint32_t getTextureHeight(const TargetCode* targetCode, uint32_t materialId, uint32_t index);
    uint32_t getTextureBytesPerTexel(const TargetCode* targetCode, uint32_t materialId, uint32_t index);

    MaterialManager();
    ~MaterialManager();
};
} // namespace oka
