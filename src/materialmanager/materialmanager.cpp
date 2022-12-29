#include "materialmanager.h"

#include "materials.h"

#include <mi/mdl_sdk.h>

#include "mdlPtxCodeGen.h"
#include "mdlLogger.h"
#include "mdlMaterialCompiler.h"
#include "mdlNeurayLoader.h"
#include "mdlRuntime.h"
#include "mtlxMdlCodeGen.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <unordered_map>

namespace fs = std::filesystem;
namespace oka
{
struct MaterialManager::Module
{
    std::string moduleName;
    std::string identifier;
};
struct MaterialManager::MaterialInstance
{
    mi::base::Handle<mi::neuraylib::IFunction_call> instance;
};
struct MaterialManager::CompiledMaterial
{
    mi::base::Handle<mi::neuraylib::ICompiled_material> compiledMaterial;
    std::string name;
};
struct MaterialManager::TargetCode
{
    struct InternalMaterial
    {
        mi::base::Handle<const mi::neuraylib::ITarget_code> targetCode;
        std::string codePtx;
        CompiledMaterial* compiledMaterial;
        mi::base::Uuid hash;
        uint32_t hash32;
        uint32_t functionNum;
        bool isUnique = false;
        mi::base::Handle<mi::neuraylib::ITarget_argument_block> arg_block;
        mi::Size argument_block_layout_index = -1;
        uint32_t arg_block_offset = -1; // offset for arg block in argBlockData
        uint32_t ro_offset = -1;
    };
    std::vector<InternalMaterial> internalMaterials;
    std::unordered_map<uint32_t, uint32_t> uidToInternalIndex;
    std::unordered_map<CompiledMaterial*, uint32_t> ptrToInternalIndex;
    std::vector<Mdl_resource_info> resourceInfo;
    std::vector<oka::MdlPtxCodeGen::InternalMaterialInfo> internalsInfo;
    std::vector<const mi::neuraylib::ICompiled_material*> compiledMaterials;
    bool isInitialized = false;

    // MDL data for GPU buffers
    std::vector<MdlMaterial> mdlMaterials;
    std::vector<uint8_t> argBlockData;
    std::vector<uint8_t> roData;
};

struct MaterialManager::TextureDescription
{
    std::string dbName;
    mi::base::Handle<const mi::neuraylib::IType_texture> textureType;
};

class Resource_callback : public mi::base::Interface_implement<mi::neuraylib::ITarget_resource_callback>
{
public:
    /// Constructor.
    Resource_callback(mi::base::Handle<mi::neuraylib::ITransaction>& transaction,
                      const mi::base::Handle<const mi::neuraylib::ITarget_code>& target_code)
        : m_transaction(transaction), m_target_code(target_code)
    {
    }

    /// Destructor.
    virtual ~Resource_callback() = default;

    /// Returns a resource index for the given resource value usable by the target code
    /// resource handler for the corresponding resource type.
    ///
    /// \param resource  the resource value
    ///
    /// \returns a resource index or 0 if no resource index can be returned
    mi::Uint32 get_resource_index(mi::neuraylib::IValue_resource const* resource) override
    {
        // check whether we already know the resource index
        auto it = m_resource_cache.find(resource);
        if (it != m_resource_cache.end())
            return it->second;

        const char* filePath = resource->get_file_path();
        // handle resources already known by the target code
        mi::Uint32 res_idx = m_target_code->get_known_resource_index(m_transaction.get(), resource);
        if (res_idx > 0)
        {
            // only accept body resources
            switch (resource->get_kind())
            {
            case mi::neuraylib::IValue::VK_TEXTURE:
                if (res_idx < m_target_code->get_body_texture_count())
                    return res_idx;
                break;
            case mi::neuraylib::IValue::VK_LIGHT_PROFILE:
                if (res_idx < m_target_code->get_body_light_profile_count())
                    return res_idx;
                break;
            case mi::neuraylib::IValue::VK_BSDF_MEASUREMENT:
                if (res_idx < m_target_code->get_body_bsdf_measurement_count())
                    return res_idx;
                break;
            default:
                return 0u; // invalid kind
            }
        }

        // invalid (or empty) resource
        const char* name = resource->get_value();
        if (!name)
        {
            return 0;
        }

        switch (resource->get_kind())
        {
        case mi::neuraylib::IValue::VK_TEXTURE: {
            mi::base::Handle<mi::neuraylib::IValue_texture const> val_texture(
                resource->get_interface<mi::neuraylib::IValue_texture const>());
            if (!val_texture)
                return 0u; // unknown resource

            mi::base::Handle<const mi::neuraylib::IType_texture> texture_type(val_texture->get_type());

            mi::neuraylib::ITarget_code::Texture_shape shape =
                mi::neuraylib::ITarget_code::Texture_shape(texture_type->get_shape());

            // m_compile_result.textures.emplace_back(resource->get_value(), shape);
            // res_idx = m_compile_result.textures.size() - 1;
            break;
        }
        case mi::neuraylib::IValue::VK_LIGHT_PROFILE:
            // m_compile_result.light_profiles.emplace_back(resource->get_value());
            // res_idx = m_compile_result.light_profiles.size() - 1;
            break;
        case mi::neuraylib::IValue::VK_BSDF_MEASUREMENT:
            // m_compile_result.bsdf_measurements.emplace_back(resource->get_value());
            // res_idx = m_compile_result.bsdf_measurements.size() - 1;
            break;
        default:
            return 0u; // invalid kind
        }

        m_resource_cache[resource] = res_idx;
        return res_idx;
    }

    /// Returns a string identifier for the given string value usable by the target code.
    ///
    /// The value 0 is always the "not known string".
    ///
    /// \param s  the string value
    mi::Uint32 get_string_index(mi::neuraylib::IValue_string const* s) override
    {
        char const* str_val = s->get_value();
        if (str_val == nullptr)
            return 0u;

        for (mi::Size i = 0, n = m_target_code->get_string_constant_count(); i < n; ++i)
        {
            if (strcmp(m_target_code->get_string_constant(i), str_val) == 0)
                return mi::Uint32(i);
        }

        // string not known by code
        return 0u;
    }

private:
    mi::base::Handle<mi::neuraylib::ITransaction> m_transaction;
    mi::base::Handle<const mi::neuraylib::ITarget_code> m_target_code;

    std::unordered_map<mi::neuraylib::IValue_resource const*, mi::Uint32> m_resource_cache;
};

class MaterialManager::Context
{
public:
    Context()
    {
        configurePaths();
    };

    ~Context(){};

    // paths is array of resource pathes + mdl path
    bool addMdlSearchPath(const char* paths[], uint32_t numPaths)
    {
        mRuntime = std::make_unique<oka::MdlRuntime>();
        if (!mRuntime->init(paths, numPaths, mPathso.c_str(), mImagePluginPath.c_str()))
        {
            return false;
        }

        mMtlxCodeGen = std::make_unique<oka::MtlxMdlCodeGen>(mtlxLibPath.c_str());

        mMatCompiler = std::make_unique<oka::MdlMaterialCompiler>(*mRuntime);
        mTransaction = mi::base::Handle<mi::neuraylib::ITransaction>(mRuntime->getTransaction());
        mNeuray = mRuntime->getNeuray();

        mCodeGen = std::make_unique<MdlPtxCodeGen>();
        mCodeGen->init(*mRuntime);
        std::vector<char> data;
        {
            const std::string cwdPath = fs::current_path().string();
            const std::string precompiledPath = cwdPath + "\\shaders\\OptixRender_radiance_closest_hit.bc";

            std::ifstream file(precompiledPath.c_str(), std::ios::in | std::ios::binary);

            if (!file.is_open())
            {
                std::cerr << "Cannot open precompiled file.\n";
                return false;
            }

            file.seekg(0, std::ios::end);
            data.resize(file.tellg());

            file.seekg(0, std::ios::beg);
            file.read(data.data(), data.size());
        }

        if (!mCodeGen->setOptionBinary("llvm_renderer_module", data.data(), data.size()))
        {
            std::cerr << "Unable to set binary options!\n";
            return false;
        }

        return true;
    }

    Module* createMtlxModule(const char* mtlSrc)
    {
        assert(mMtlxCodeGen);
        std::unique_ptr<Module> module = std::make_unique<Module>();

        bool res = mMtlxCodeGen->translate(mtlSrc, mMdlSrc, module->identifier);
        if (res)
        {
            std::string mdlFile = "./data/materials/mtlx/" + module->identifier + ".mdl";
            std::ofstream material(mdlFile.c_str());
            if (!material.is_open())
            {
                printf("Error: can not create file: %s\n", mdlFile.c_str());
                return nullptr;
            }
            material << mMdlSrc;
            material.close();
            if (!mMatCompiler->createModule(module->identifier, module->moduleName))
            {
                printf("Error: failed to create MDL module\n");
                return nullptr;
            }
            return module.release();
        }
        else
        {
            printf("Error: failed to translate MaterialX -> MDL\n");
            return nullptr;
        }
    };

    Module* createModule(const char* file)
    {
        std::unique_ptr<Module> module = std::make_unique<Module>();

        const fs::path materialFile = file;
        module->identifier = materialFile.stem().string();

        if (!mMatCompiler->createModule(module->identifier, module->moduleName))
        {
            return nullptr;
        }

        return module.release();
    };

    void destroyModule(Module* module)
    {
        assert(module);
        delete module;
    };

    MaterialInstance* createMaterialInstance(MaterialManager::Module* module, const char* materialName)
    {
        assert(module);
        assert(materialName);

        std::unique_ptr<MaterialInstance> material = std::make_unique<MaterialInstance>();

        if (strcmp(materialName, "") == 0) // mtlx
        {
            materialName = module->identifier.c_str();
        }

        if (!mMatCompiler->createMaterialInstace(module->moduleName.c_str(), materialName, material->instance))
        {
            return nullptr;
        }

        return material.release();
    }

    void destroyMaterialInstance(MaterialInstance* matInst)
    {
        assert(matInst);
        delete matInst;
    }

    void dumpParams(const TargetCode* targetCode, CompiledMaterial* material)
    {
        assert(targetCode);
        assert(targetCode->isInitialized);
        assert(material);
        const uint32_t internalIndex = targetCode->ptrToInternalIndex.at(material);
        const mi::Size argLayoutIndex = targetCode->internalMaterials[internalIndex].argument_block_layout_index;
        mi::base::Handle<const mi::neuraylib::ITarget_value_layout> arg_layout(
            targetCode->internalMaterials[internalIndex].targetCode->get_argument_block_layout(argLayoutIndex));

        if (targetCode->internalMaterials[internalIndex].arg_block == nullptr)
        {
            printf("Material : %s has no arg block!\n", material->name.c_str());
            return;
        }

        char* baseArgBlockData = targetCode->internalMaterials[internalIndex].arg_block->get_data();
        const mi::Size argBlockOffset = targetCode->internalMaterials[internalIndex].arg_block_offset;

        const mi::Size paramCount = material->compiledMaterial->get_parameter_count();
        printf("Material name: %s\t param count: %lld\n", material->name.c_str(), paramCount);
        for (int pi = 0; pi < paramCount; ++pi)
        {
            const char* name = material->compiledMaterial->get_parameter_name(pi);
            printf("# %d : Param name: %s\n", pi, name);
            mi::neuraylib::Target_value_layout_state valLayoutState = arg_layout->get_nested_state(pi);
            mi::neuraylib::IValue::Kind kind;
            mi::Size arg_size;
            const mi::Size offsetInArgBlock = arg_layout->get_layout(kind, arg_size, valLayoutState);
            printf("\t offset = %llu\t size = %llu\n", offsetInArgBlock, arg_size);

            // char* data = baseArgBlockData + offsetInArgBlock;
            const uint8_t* data = targetCode->argBlockData.data() + argBlockOffset + offsetInArgBlock;
            switch (kind)
            {
            case mi::neuraylib::IValue::Kind::VK_FLOAT: {
                float val = *((float*)data);
                printf("\t type float\n");
                printf("\t val = %f\n", val);
                break;
            }
            case mi::neuraylib::IValue::Kind::VK_INT: {
                int val = *((int*)data);
                printf("\t type int\n");
                printf("\t val = %d\n", val);
                break;
            }
            case mi::neuraylib::IValue::Kind::VK_BOOL: {
                bool val = *((bool*)data);
                printf("\t type bool\n");
                printf("\t val = %d\n", val);
                break;
            }
            case mi::neuraylib::IValue::Kind::VK_COLOR: {
                float* val = (float*)data;
                printf("\t type color\n");
                printf("\t val = (%f, %f, %f)\n", val[0], val[1], val[2]);
                break;
            }
            case mi::neuraylib::IValue::Kind::VK_TEXTURE: {
                printf("\t type texture\n");
                int val = *((int*)data);
                printf("\t val = %d\n", val);
                break;
            }

            default:
                break;
            }

            // uint8_t* dst = targetCode->argBlockData.data() + argBlockOffset + offsetInArgBlock;
            // memcpy(dst, param.value.data(), param.value.size());
            // isParamFound = true;
        }
    }

    bool setParam(TargetCode* targetCode, CompiledMaterial* material, const Param& param)
    {
        assert(targetCode);
        assert(targetCode->isInitialized);
        assert(material);
        bool isParamFound = false;
        for (int pi = 0; pi < material->compiledMaterial->get_parameter_count(); ++pi)
        {
            if (!strcmp(material->compiledMaterial->get_parameter_name(pi), param.name.c_str()))
            {
                const uint32_t internalIndex = targetCode->ptrToInternalIndex[material];
                const mi::Size argLayoutIndex = targetCode->internalMaterials[internalIndex].argument_block_layout_index;
                mi::base::Handle<const mi::neuraylib::ITarget_value_layout> arg_layout(
                    targetCode->internalMaterials[internalIndex].targetCode->get_argument_block_layout(argLayoutIndex));
                mi::neuraylib::Target_value_layout_state valLayoutState = arg_layout->get_nested_state(pi);
                mi::neuraylib::IValue::Kind kind;
                mi::Size arg_size;
                const mi::Size offsetInArgBlock = arg_layout->get_layout(kind, arg_size, valLayoutState);
                assert(arg_size == param.value.size()); // TODO: should match, otherwise error
                const mi::Size argBlockOffset = targetCode->internalMaterials[internalIndex].arg_block_offset;
                uint8_t* dst = targetCode->argBlockData.data() + argBlockOffset + offsetInArgBlock;
                memcpy(dst, param.value.data(), param.value.size());
                isParamFound = true;
                break;
            }
        }
        return isParamFound;
    }

    TextureDescription* createTextureDescription(const char* name, const char* gammaMode)
    {
        assert(name);
        float gamma = 1.0;
        const char* texGamma = "";
        if (strcmp(gammaMode, "srgb") == 0)
        {
            gamma = 2.2f;
            texGamma = "_srgb";
        }
        else if (strcmp(gammaMode, "linear") == 0)
        {
            gamma = 1.0f;
            texGamma = "_linear";
        }

        std::string textureDbName = std::string(name) + std::string(texGamma);
        mi::base::Handle<const mi::neuraylib::ITexture> texAccess = mi::base::Handle<const mi::neuraylib::ITexture>(
            mMatCompiler->getTransaction()->access<mi::neuraylib::ITexture>(textureDbName.c_str()));

        // check if it is in DB
        if (!texAccess.is_valid_interface())
        {
            // Load it
            mi::base::Handle<mi::neuraylib::ITexture> tex(
                mMatCompiler->getTransaction()->create<mi::neuraylib::ITexture>("Texture"));
            mi::base::Handle<mi::neuraylib::IImage> image(
                mMatCompiler->getTransaction()->create<mi::neuraylib::IImage>("Image"));
            if (image->reset_file(name) != 0)
            {
                // TODO: report error could not find texture!
                return nullptr;
            }
            std::string imageName = textureDbName + std::string("_image");
            mMatCompiler->getTransaction()->store(image.get(), imageName.c_str());

            tex->set_image(imageName.c_str());
            tex->set_gamma(gamma);
            mMatCompiler->getTransaction()->store(tex.get(), textureDbName.c_str());
        }

        mi::base::Handle<mi::neuraylib::IType_factory> typeFactory(
            mMatCompiler->getFactory()->create_type_factory(mMatCompiler->getTransaction().get()));
        // TODO: texture could be 1D, 3D
        mi::base::Handle<const mi::neuraylib::IType_texture> textureType(
            typeFactory->create_texture(mi::neuraylib::IType_texture::TS_2D));

        TextureDescription* texDesc = new TextureDescription;
        texDesc->dbName = textureDbName;
        texDesc->textureType = textureType;
        return texDesc;
    }

    const char* getTextureDbName(TextureDescription* texDesc)
    {
        return texDesc->dbName.c_str();
    }

    CompiledMaterial* compileMaterial(MaterialInstance* matInstance)
    {
        assert(matInstance);
        std::unique_ptr<CompiledMaterial> material = std::make_unique<CompiledMaterial>();
        if (!mMatCompiler->compileMaterial(matInstance->instance, material->compiledMaterial))
        {
            return nullptr;
        }
        material->name = matInstance->instance->get_function_definition();

        return material.release();
    };

    void destroyCompiledMaterial(CompiledMaterial* materials)
    {
        assert(materials);
        delete materials;
    }

    const char* getName(CompiledMaterial* compMaterial)
    {
        return compMaterial->name.c_str();
    }

    TargetCode* generateTargetCode(CompiledMaterial** materials, const uint32_t numMaterials)
    {
        TargetCode* targetCode = new TargetCode;

        std::unordered_map<uint32_t, uint32_t> uidToFunctionNum;
        std::vector<TargetCode::InternalMaterial>& internalMaterials = targetCode->internalMaterials;
        internalMaterials.resize(numMaterials);

        std::vector<const mi::neuraylib::ICompiled_material*> materialsToCompile;
        for (int i = 0; i < numMaterials; ++i)
        {
            internalMaterials[i].compiledMaterial = materials[i];
            internalMaterials[i].hash = materials[i]->compiledMaterial->get_hash();
            internalMaterials[i].hash32 = uuid_hash32(internalMaterials[i].hash);
            uint32_t functionNum = -1;
            bool isUnique = false;
            if (uidToFunctionNum.find(internalMaterials[i].hash32) != uidToFunctionNum.end())
            {
                functionNum = uidToFunctionNum[internalMaterials[i].hash32];
            }
            else
            {
                functionNum = uidToFunctionNum.size();
                uidToFunctionNum[internalMaterials[i].hash32] = functionNum;
                isUnique = true;
                materialsToCompile.push_back(internalMaterials[i].compiledMaterial->compiledMaterial.get());
                // log output
                printf("Material to compile: %s\n", getName(internalMaterials[i].compiledMaterial));
            }
            internalMaterials[i].isUnique = isUnique;
            internalMaterials[i].functionNum = functionNum;
            targetCode->uidToInternalIndex[internalMaterials[i].hash32] = i;
            targetCode->ptrToInternalIndex[materials[i]] = i;
        }

        const uint32_t numMaterialsToCompile = materialsToCompile.size();
        printf("Num Materials to compile: %d\n", numMaterialsToCompile);
        std::vector<oka::MaterialManager::TargetCode::InternalMaterial> compiledInternalMaterials(numMaterialsToCompile);
        for (int i = 0; i < numMaterialsToCompile; ++i)
        {
            oka::MdlPtxCodeGen::InternalMaterialInfo internalsInfo;
            compiledInternalMaterials[i].targetCode =
                mCodeGen->translate(materialsToCompile[i], compiledInternalMaterials[i].codePtx, internalsInfo);
            compiledInternalMaterials[i].argument_block_layout_index = internalsInfo.argument_block_index;
        }

        targetCode->mdlMaterials.resize(numMaterials);
        for (uint32_t i = 0; i < numMaterials; ++i)
        {
            const uint32_t compiledOrder = internalMaterials[i].functionNum;
            assert(compiledOrder < numMaterialsToCompile);
            internalMaterials[i].argument_block_layout_index = compiledInternalMaterials[compiledOrder].argument_block_layout_index;
            internalMaterials[i].targetCode = compiledInternalMaterials[compiledOrder].targetCode;
            internalMaterials[i].codePtx = compiledInternalMaterials[compiledOrder].codePtx;
            targetCode->mdlMaterials[i].functionId = compiledOrder;
        }

        targetCode->argBlockData = loadArgBlocks(targetCode);
        targetCode->roData = loadROData(targetCode);

        // uint32_t texCount = targetCode->targetCode->get_texture_count();
        // if (texCount > 0)
        // {
        //     targetCode->resourceInfo.resize(texCount);
        //     for (uint32_t i = 0; i < texCount; ++i)
        //     {
        //         targetCode->resourceInfo[i].gpu_resource_array_start = i;
        //         const char* texUrl = targetCode->targetCode->get_texture_url(i);
        //         const int texBody = targetCode->targetCode->get_body_texture_count();

        //         const char* texName = getTextureName(targetCode, i);
        //         // const float* data = getTextureData(targetCode, i);
        //         // uint32_t width = getTextureWidth(targetCode, i);
        //         // uint32_t height = getTextureHeight(targetCode, i);
        //         // const char* type = getTextureType(targetCode, i);
        //         printf("Material texture name: %s\n", texName);
        //     }
        // }
        // else
        //{
        //     targetCode->resourceInfo.resize(1);
        // }

        targetCode->isInitialized = true;
        return targetCode;
    };

    int registerResource(TargetCode* targetCode, int index)
    {
        Mdl_resource_info ri{ 0 };
        ri.gpu_resource_array_start = index;
        targetCode->resourceInfo.push_back(ri);
        return (int)targetCode->resourceInfo.size(); // resource id 0 is reserved for invalid, but we store resources
                                                     // from 0 internally
    }

    const char* getShaderCode(const TargetCode* targetCode, uint32_t materialIdx)
    {
        const uint32_t compiledOrder = targetCode->internalMaterials[materialIdx].functionNum;
        return targetCode->internalMaterials[compiledOrder].codePtx.c_str();
    }

    uint32_t getArgBlockOffset(const TargetCode* targetCode, uint32_t materialId)
    {
        return targetCode->internalMaterials[materialId].arg_block_offset;
    }

    uint32_t getReadOnlyBlockOffset(const TargetCode* targetCode, uint32_t materialId)
    {
        return targetCode->internalMaterials[materialId].ro_offset;
    }

    uint32_t getReadOnlyBlockSize(const TargetCode* shaderCode)
    {
        return shaderCode->roData.size();
    }

    const uint8_t* getReadOnlyBlockData(const TargetCode* targetCode)
    {
        return targetCode->roData.data();
    }

    uint32_t getArgBufferSize(const TargetCode* shaderCode)
    {
        return shaderCode->argBlockData.size();
    }
    const uint8_t* getArgBufferData(const TargetCode* targetCode)
    {
        return targetCode->argBlockData.data();
    }

    uint32_t getResourceInfoSize(const TargetCode* targetCode)
    {
        return targetCode->resourceInfo.size() * sizeof(Mdl_resource_info);
    }

    const uint8_t* getResourceInfoData(const TargetCode* targetCode)
    {
        return reinterpret_cast<const uint8_t*>(targetCode->resourceInfo.data());
    }

    uint32_t getMdlMaterialSize(const TargetCode* targetCode)
    {
        return targetCode->mdlMaterials.size() * sizeof(MdlMaterial);
    }

    const uint8_t* getMdlMaterialData(const TargetCode* targetCode)
    {
        return reinterpret_cast<const uint8_t*>(targetCode->mdlMaterials.data());
    }

    uint32_t getTextureCount(const TargetCode* targetCode, uint32_t materialId)
    {
        return targetCode->internalMaterials[materialId].targetCode->get_texture_count();
    }

    const char* getTextureName(const TargetCode* targetCode, uint32_t materialId, uint32_t index)
    {
        assert(index); // index == 0 is invalid
        return targetCode->internalMaterials[materialId].targetCode->get_texture(index);
    }

    const float* getTextureData(const TargetCode* targetCode, uint32_t materialId, uint32_t index)
    {
        // assert(index); // index == 0 is invalid
        assert(mTransaction);

        auto cachedCanvas = m_indexToCanvas.find(index);

        if (cachedCanvas == m_indexToCanvas.end())
        {

            mi::base::Handle<mi::neuraylib::IImage_api> image_api(mNeuray->get_api_component<mi::neuraylib::IImage_api>());
            mi::base::Handle<const mi::neuraylib::ITexture> texture(mTransaction->access<mi::neuraylib::ITexture>(
                targetCode->internalMaterials[materialId].targetCode->get_texture(index)));
            mi::base::Handle<const mi::neuraylib::IImage> image(
                mTransaction->access<mi::neuraylib::IImage>(texture->get_image()));
            mi::base::Handle<const mi::neuraylib::ICanvas> canvas(image->get_canvas(0, 0, 0));
            char const* image_type = image->get_type(0, 0);

            // if (canvas->get_tiles_size_x() != 1 || canvas->get_tiles_size_y() != 1)
            // {
            //     mLogger->message(mi::base::MESSAGE_SEVERITY_ERROR, "The example does not support tiled images!");
            //     return nullptr;
            // }

            if (texture->get_effective_gamma(0, 0) != 1.0f)
            {
                // Copy/convert to float4 canvas and adjust gamma from "effective gamma" to 1.
                mi::base::Handle<mi::neuraylib::ICanvas> gamma_canvas(image_api->convert(canvas.get(), "Color"));
                gamma_canvas->set_gamma(texture->get_effective_gamma(0, 0));
                image_api->adjust_gamma(gamma_canvas.get(), 1.0f);
                canvas = gamma_canvas;
            }
            else if (strcmp(image_type, "Color") != 0 && strcmp(image_type, "Float32<4>") != 0)
            {
                // Convert to expected format
                canvas = image_api->convert(canvas.get(), "Color");
            }
            m_indexToCanvas[index] = canvas;
            cachedCanvas = m_indexToCanvas.find(index);
        }

        mi::Float32 const* data = nullptr;
        mi::neuraylib::ITarget_code::Texture_shape texture_shape =
            targetCode->internalMaterials[materialId].targetCode->get_texture_shape(index);
        if (texture_shape == mi::neuraylib::ITarget_code::Texture_shape_2d)
        {
            mi::base::Handle<const mi::neuraylib::ITile> tile(cachedCanvas->second->get_tile());
            data = static_cast<mi::Float32 const*>(tile->get_data());
        }

        return data;
    }

    const char* getTextureType(const TargetCode* targetCode, uint32_t materialId, uint32_t index)
    {
        // assert(index); // index == 0 is invalid
        mi::base::Handle<const mi::neuraylib::ITexture> texture(mTransaction->access<mi::neuraylib::ITexture>(
            targetCode->internalMaterials[materialId].targetCode->get_texture(index)));
        mi::base::Handle<const mi::neuraylib::IImage> image(
            mTransaction->access<mi::neuraylib::IImage>(texture->get_image()));
        char const* imageType = image->get_type(0, 0);

        return imageType;
    }

    uint32_t getTextureWidth(const TargetCode* targetCode, uint32_t materialId, uint32_t index)
    {
        // assert(index); // index == 0 is invalid
        mi::base::Handle<const mi::neuraylib::ITexture> texture(mTransaction->access<mi::neuraylib::ITexture>(
            targetCode->internalMaterials[materialId].targetCode->get_texture(index)));
        mi::base::Handle<const mi::neuraylib::IImage> image(
            mTransaction->access<mi::neuraylib::IImage>(texture->get_image()));
        mi::base::Handle<const mi::neuraylib::ICanvas> canvas(image->get_canvas(0, 0, 0));
        mi::Uint32 texWidth = canvas->get_resolution_x();

        return texWidth;
    }

    uint32_t getTextureHeight(const TargetCode* targetCode, uint32_t materialId, uint32_t index)
    {
        // assert(index); // index == 0 is invalid
        mi::base::Handle<const mi::neuraylib::ITexture> texture(mTransaction->access<mi::neuraylib::ITexture>(
            targetCode->internalMaterials[materialId].targetCode->get_texture(index)));
        mi::base::Handle<const mi::neuraylib::IImage> image(
            mTransaction->access<mi::neuraylib::IImage>(texture->get_image()));
        mi::base::Handle<const mi::neuraylib::ICanvas> canvas(image->get_canvas(0, 0, 0));
        mi::Uint32 texHeight = canvas->get_resolution_y();

        return texHeight;
    }

    uint32_t getTextureBytesPerTexel(const TargetCode* targetCode, uint32_t materialId, uint32_t index)
    {
        // assert(index); // index == 0 is invalid
        mi::base::Handle<mi::neuraylib::IImage_api> image_api(mNeuray->get_api_component<mi::neuraylib::IImage_api>());

        mi::base::Handle<const mi::neuraylib::ITexture> texture(mTransaction->access<mi::neuraylib::ITexture>(
            targetCode->internalMaterials[materialId].targetCode->get_texture(index)));
        mi::base::Handle<const mi::neuraylib::IImage> image(
            mTransaction->access<mi::neuraylib::IImage>(texture->get_image()));
        char const* imageType = image->get_type(0, 0);

        int cpp = image_api->get_components_per_pixel(imageType);
        int bpc = image_api->get_bytes_per_component(imageType);
        int bpp = cpp * bpc;

        return bpp;
    }

private:
    std::string mImagePluginPath;
    std::string mPathso;
    std::string mMdlSrc;
    std::string mtlxLibPath;

    std::unordered_map<uint32_t, mi::base::Handle<const mi::neuraylib::ICanvas>> m_indexToCanvas;

    void configurePaths()
    {
        using namespace std;
        const fs::path cwd = fs::current_path();
        mtlxLibPath = "";
        const char* envUSDPath = std::getenv("USD_DIR");
        if (!envUSDPath)
        {
            printf("Please, set USD_DIR variable\n");
            assert(0);
        }
        else
        {
            mtlxLibPath = std::string(envUSDPath) + "./libraries/";
        }
        mMdlSrc = cwd.string() + "/data/materials/mtlx"; // path to the material

#ifdef MI_PLATFORM_WINDOWS
        mPathso = cwd.string();
        mImagePluginPath = cwd.string() + "/nv_freeimage.dll";
#else
        mPathso = cwd.string();
        mImagePluginPath = cwd.string() + "/nv_freeimage.so";
#endif
    }

    std::unique_ptr<oka::MdlPtxCodeGen> mCodeGen = nullptr;
    std::unique_ptr<oka::MdlMaterialCompiler> mMatCompiler = nullptr;
    std::unique_ptr<oka::MdlRuntime> mRuntime = nullptr;
    std::unique_ptr<oka::MtlxMdlCodeGen> mMtlxCodeGen = nullptr;

    mi::base::Handle<mi::neuraylib::ITransaction> mTransaction;
    mi::base::Handle<oka::MdlLogger> mLogger;
    mi::base::Handle<mi::neuraylib::INeuray> mNeuray;

    std::vector<uint8_t> loadArgBlocks(TargetCode* targetCode);
    std::vector<uint8_t> loadROData(TargetCode* targetCode);
};

MaterialManager::MaterialManager()
{
    mContext = std::make_unique<Context>();
}

MaterialManager::~MaterialManager()
{
    mContext.reset(nullptr);
};

bool MaterialManager::addMdlSearchPath(const char* paths[], uint32_t numPaths)
{
    return mContext->addMdlSearchPath(paths, numPaths);
}

MaterialManager::Module* MaterialManager::createModule(const char* file)
{
    return mContext->createModule(file);
}

MaterialManager::Module* MaterialManager::createMtlxModule(const char* file)
{
    return mContext->createMtlxModule(file);
}

void MaterialManager::destroyModule(MaterialManager::Module* module)
{
    return mContext->destroyModule(module);
}

MaterialManager::MaterialInstance* MaterialManager::createMaterialInstance(MaterialManager::Module* module,
                                                                           const char* materialName)
{
    return mContext->createMaterialInstance(module, materialName);
}

void MaterialManager::destroyMaterialInstance(MaterialManager::MaterialInstance* matInst)
{
    return mContext->destroyMaterialInstance(matInst);
}

MaterialManager::TextureDescription* MaterialManager::createTextureDescription(const char* name, const char* gamma)
{
    return mContext->createTextureDescription(name, gamma);
}

const char* MaterialManager::getTextureDbName(TextureDescription* texDesc)
{
    return mContext->getTextureDbName(texDesc);
}

void MaterialManager::dumpParams(const TargetCode* targetCode, CompiledMaterial* material)
{
    return mContext->dumpParams(targetCode, material);
}

bool MaterialManager::setParam(TargetCode* targetCode, CompiledMaterial* material, const Param& param)
{
    return mContext->setParam(targetCode, material, param);
}

MaterialManager::CompiledMaterial* MaterialManager::compileMaterial(MaterialManager::MaterialInstance* matInstance)
{
    return mContext->compileMaterial(matInstance);
}

void MaterialManager::destroyCompiledMaterial(MaterialManager::CompiledMaterial* material)
{
    return mContext->destroyCompiledMaterial(material);
}

const char* MaterialManager::getName(CompiledMaterial* compMaterial)
{
    return mContext->getName(compMaterial);
}

MaterialManager::TargetCode* MaterialManager::generateTargetCode(CompiledMaterial** materials, uint32_t numMaterials)
{
    return mContext->generateTargetCode(materials, numMaterials);
}

const char* MaterialManager::getShaderCode(const TargetCode* targetCode, uint32_t materialId) // get shader code
{
    return mContext->getShaderCode(targetCode, materialId);
}

uint32_t MaterialManager::getReadOnlyBlockSize(const TargetCode* targetCode)
{
    return mContext->getReadOnlyBlockSize(targetCode);
}

const uint8_t* MaterialManager::getReadOnlyBlockData(const TargetCode* targetCode)
{
    return mContext->getReadOnlyBlockData(targetCode);
}

uint32_t MaterialManager::getArgBufferSize(const TargetCode* targetCode)
{
    return mContext->getArgBufferSize(targetCode);
}

const uint8_t* MaterialManager::getArgBufferData(const TargetCode* targetCode)
{
    return mContext->getArgBufferData(targetCode);
}

uint32_t MaterialManager::getArgBlockOffset(const TargetCode* targetCode, uint32_t materialId)
{
    return mContext->getArgBlockOffset(targetCode, materialId);
}

uint32_t MaterialManager::getReadOnlyOffset(const TargetCode* targetCode, uint32_t materialId)
{
    return mContext->getArgBlockOffset(targetCode, materialId);
}

uint32_t MaterialManager::getResourceInfoSize(const TargetCode* targetCode)
{
    return mContext->getResourceInfoSize(targetCode);
}

const uint8_t* MaterialManager::getResourceInfoData(const TargetCode* targetCode)
{
    return mContext->getResourceInfoData(targetCode);
}

int MaterialManager::registerResource(TargetCode* targetCode, int index)
{
    return mContext->registerResource(targetCode, index);
}

uint32_t MaterialManager::getMdlMaterialSize(const TargetCode* targetCode)
{
    return mContext->getMdlMaterialSize(targetCode);
}

const uint8_t* MaterialManager::getMdlMaterialData(const TargetCode* targetCode)
{
    return mContext->getMdlMaterialData(targetCode);
}

uint32_t MaterialManager::getTextureCount(const TargetCode* targetCode, uint32_t materialId)
{
    return mContext->getTextureCount(targetCode, materialId);
}

const char* MaterialManager::getTextureName(const TargetCode* targetCode, uint32_t materialId, uint32_t index)
{
    return mContext->getTextureName(targetCode, materialId, index);
}

const float* MaterialManager::getTextureData(const TargetCode* targetCode, uint32_t materialId, uint32_t index)
{
    return mContext->getTextureData(targetCode, materialId, index);
}

const char* MaterialManager::getTextureType(const TargetCode* targetCode, uint32_t materialId, uint32_t index)
{
    return mContext->getTextureType(targetCode, materialId, index);
}

uint32_t MaterialManager::getTextureWidth(const TargetCode* targetCode, uint32_t materialId, uint32_t index)
{
    return mContext->getTextureWidth(targetCode, materialId, index);
}

uint32_t MaterialManager::getTextureHeight(const TargetCode* targetCode, uint32_t materialId, uint32_t index)
{
    return mContext->getTextureHeight(targetCode, materialId, index);
}

uint32_t MaterialManager::getTextureBytesPerTexel(const TargetCode* targetCode, uint32_t materialId, uint32_t index)
{
    return mContext->getTextureBytesPerTexel(targetCode, materialId, index);
}

inline size_t round_to_power_of_two(size_t value, size_t power_of_two_factor)
{
    return (value + (power_of_two_factor - 1)) & ~(power_of_two_factor - 1);
}

std::vector<uint8_t> MaterialManager::Context::loadArgBlocks(TargetCode* targetCode)
{
    std::vector<uint8_t> res;
    for (int i = 0; i < targetCode->internalMaterials.size(); ++i)
    {
        mi::base::Handle<const mi::neuraylib::ITarget_code>& mdlTargetCode = targetCode->internalMaterials[i].targetCode;
        mi::base::Handle<Resource_callback> callback(new Resource_callback(mTransaction, mdlTargetCode));

        // const uint32_t compiledIndex = targetCode->internalMaterials[i].functionNum;
        const mi::Size argLayoutIndex = targetCode->internalMaterials[i].argument_block_layout_index;
        const mi::Size layoutCount = mdlTargetCode->get_argument_layout_count();
        if (argLayoutIndex != static_cast<mi::Size>(-1) && argLayoutIndex < layoutCount)
        {
            mi::neuraylib::ICompiled_material* compiledMaterial =
                targetCode->internalMaterials[i].compiledMaterial->compiledMaterial.get();
            targetCode->internalMaterials[i].arg_block =
                mdlTargetCode->create_argument_block(argLayoutIndex, compiledMaterial, callback.get());

            if (!targetCode->internalMaterials[i].arg_block)
            {
                std::cerr << ("Failed to create material argument block: ") << std::endl;
                res.resize(4);
                return res;
            }
            // create a buffer to provide those parameters to the shader
            // align to 4 bytes and pow of two
            const size_t buffer_size = round_to_power_of_two(targetCode->internalMaterials[i].arg_block->get_size(), 4);
            std::vector<uint8_t> argBlockData = std::vector<uint8_t>(buffer_size, 0);
            memcpy(argBlockData.data(), targetCode->internalMaterials[i].arg_block->get_data(),
                   targetCode->internalMaterials[i].arg_block->get_size());
            // set offset in common arg block buffer
            targetCode->internalMaterials[i].arg_block_offset = (uint32_t)res.size();
            targetCode->mdlMaterials[i].arg_block_offset = (int)res.size();
            res.insert(res.end(), argBlockData.begin(), argBlockData.end());
        }
    }

    if (res.empty())
    {
        res.resize(4);
    }
    return res;
}

std::vector<uint8_t> MaterialManager::Context::loadROData(TargetCode* targetCode)
{
    std::vector<uint8_t> roData;

    for (int i = 0; i < targetCode->internalMaterials.size(); ++i)
    {
        const size_t segCount = targetCode->internalMaterials[i].targetCode->get_ro_data_segment_count();
        targetCode->internalMaterials[i].ro_offset = roData.size();
        for (size_t s = 0; s < segCount; ++s)
        {
            const char* data = targetCode->internalMaterials[i].targetCode->get_ro_data_segment_data(s);
            size_t dataSize = targetCode->internalMaterials[i].targetCode->get_ro_data_segment_size(s);
            const char* name = targetCode->internalMaterials[i].targetCode->get_ro_data_segment_name(s);
            std::cerr << "MDL segment [" << s << "] name : " << name << std::endl;

            if (dataSize != 0)
            {
                roData.insert(roData.end(), data, data + dataSize);
            }
        }
    }
    if (roData.empty())
    {
        roData.resize(4);
    }

    return roData;
}
} // namespace oka
