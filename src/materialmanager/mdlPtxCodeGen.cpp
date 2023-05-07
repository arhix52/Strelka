#include "mdlPtxCodeGen.h"

#include "materials.h"

#include <mi/mdl_sdk.h>

#include <cassert>
#include <sstream>
#include <log.h>

namespace oka
{
const char* SCATTERING_FUNC_NAME = "mdl_bsdf_scattering";
const char* EMISSION_FUNC_NAME = "mdl_edf_emission";
const char* EMISSION_INTENSITY_FUNC_NAME = "mdl_edf_emission_intensity";
const char* MATERIAL_STATE_NAME = "Shading_state_material";

bool MdlPtxCodeGen::init(MdlRuntime& runtime)
{
    mi::base::Handle<mi::neuraylib::IMdl_backend_api> backendApi(runtime.getBackendApi());
    mBackend = mi::base::Handle<mi::neuraylib::IMdl_backend>(
        backendApi->get_backend(mi::neuraylib::IMdl_backend_api::MB_CUDA_PTX));
    if (!mBackend.is_valid_interface())
    {
        mLogger->message(mi::base::MESSAGE_SEVERITY_FATAL, "CUDA backend not supported by MDL runtime");
        return false;
    }
    // 75 - Turing
    // 86 - Ampere
    // 89 - Ada
    if (mBackend->set_option("sm_version", "75") != 0)
    {
        mLogger->message(mi::base::MESSAGE_SEVERITY_FATAL, "ERROR: Setting PTX option sm_version failed");
        return false;
    }
    if (mBackend->set_option("num_texture_spaces", "2") != 0)
    {
        mLogger->message(mi::base::MESSAGE_SEVERITY_FATAL, "ERROR: Setting PTX option num_texture_spaces failed");
        return false;
    }
    if (mBackend->set_option("num_texture_results", "16") != 0)
    {
        mLogger->message(mi::base::MESSAGE_SEVERITY_FATAL, "ERROR: Setting PTX option num_texture_results failed");
        return false;
    }
    if (mBackend->set_option("tex_lookup_call_mode", "direct_call") != 0)
    {
        mLogger->message(mi::base::MESSAGE_SEVERITY_FATAL, "ERROR: Setting PTX option tex_lookup_call_mode failed");
        return false;
    }

    mLogger = mi::base::Handle<MdlLogger>(runtime.getLogger());

    mLoader = std::move(runtime.mLoader);
    mi::base::Handle<mi::neuraylib::IMdl_factory> factory(runtime.getFactory());
    mContext = mi::base::Handle<mi::neuraylib::IMdl_execution_context>(factory->create_execution_context());

    mDatabase = mi::base::Handle<mi::neuraylib::IDatabase>(runtime.getDatabase());
    mTransaction = mi::base::Handle<mi::neuraylib::ITransaction>(runtime.getTransaction());
    return true;
}

mi::base::Handle<const mi::neuraylib::ITarget_code> MdlPtxCodeGen::translate(
    const mi::neuraylib::ICompiled_material* material, std::string& ptxSrc, InternalMaterialInfo& internalsInfo)
{
    assert(material);
    mi::base::Handle<mi::neuraylib::ILink_unit> linkUnit(mBackend->create_link_unit(mTransaction.get(), mContext.get()));
    mLogger->flushContextMessages(mContext.get());

    if (!linkUnit)
    {
        throw "Failed to create link unit";
    }
    mi::Size argBlockIndex;
    if (!appendMaterialToLinkUnit(0, material, linkUnit.get(), argBlockIndex))
    {
        throw "Failed to append material to the link unit";
    }
    internalsInfo.argument_block_index = argBlockIndex;
    mi::base::Handle<const mi::neuraylib::ITarget_code> targetCode(
        mBackend->translate_link_unit(linkUnit.get(), mContext.get()));
    mLogger->flushContextMessages(mContext.get());
    if (!targetCode)
    {
        throw "No target code";
    }
    ptxSrc = targetCode->get_code();
    return targetCode;
}

mi::base::Handle<const mi::neuraylib::ITarget_code> MdlPtxCodeGen::translate(
    const std::vector<const mi::neuraylib::ICompiled_material*>& materials,
    std::string& ptxSrc,
    std::vector<InternalMaterialInfo>& internalsInfo)
{
    mi::base::Handle<mi::neuraylib::ILink_unit> linkUnit(mBackend->create_link_unit(mTransaction.get(), mContext.get()));
    mLogger->flushContextMessages(mContext.get());

    if (!linkUnit)
    {
        throw "Failed to create link unit";
    }

    uint32_t materialCount = materials.size();
    internalsInfo.resize(materialCount);
    mi::Size argBlockIndex;

    for (uint32_t i = 0; i < materialCount; i++)
    {
        const mi::neuraylib::ICompiled_material* material = materials.at(i);
        assert(material);

        if (!appendMaterialToLinkUnit(i, material, linkUnit.get(), argBlockIndex))
        {
            throw "Failed to append material to the link unit";
        }

        internalsInfo[i].argument_block_index = argBlockIndex;
    }

    mi::base::Handle<const mi::neuraylib::ITarget_code> targetCode(
        mBackend->translate_link_unit(linkUnit.get(), mContext.get()));
    mLogger->flushContextMessages(mContext.get());

    if (!targetCode)
    {
        throw "No target code";
    }
    ptxSrc = targetCode->get_code();
    return targetCode;
}

bool MdlPtxCodeGen::appendMaterialToLinkUnit(uint32_t idx,
                                             const mi::neuraylib::ICompiled_material* compiledMaterial,
                                             mi::neuraylib::ILink_unit* linkUnit,
                                             mi::Size& argBlockIndex)
{
    std::string idxStr = std::to_string(idx);
    auto scatteringFuncName = std::string("mdlcode");
    auto emissionFuncName = std::string(EMISSION_FUNC_NAME) + "_" + idxStr;
    auto emissionIntensityFuncName = std::string(EMISSION_INTENSITY_FUNC_NAME) + "_" + idxStr;

    // Here we need to detect if current material for hair, if so, replace name in function description
    mi::base::Handle<mi::neuraylib::IExpression const> hairExpr(compiledMaterial->lookup_sub_expression("hair"));
    bool isHair = false;
    if (hairExpr != nullptr)
    {
        if (hairExpr->get_kind() != mi::neuraylib::IExpression::EK_CONSTANT)
        {
            isHair = true;
        }
    }

    std::vector<mi::neuraylib::Target_function_description> genFunctions;
    genFunctions.push_back(mi::neuraylib::Target_function_description( isHair ? "hair" : "surface.scattering", scatteringFuncName.c_str()));
    genFunctions.push_back(
        mi::neuraylib::Target_function_description("surface.emission.emission", emissionFuncName.c_str()));
    genFunctions.push_back(
        mi::neuraylib::Target_function_description("surface.emission.intensity", emissionIntensityFuncName.c_str()));

    mi::Sint32 result =
        linkUnit->add_material(compiledMaterial, genFunctions.data(), genFunctions.size(), mContext.get());

    mLogger->flushContextMessages(mContext.get());

    if (result == 0)
    {
        argBlockIndex = genFunctions[0].argument_block_index;
    }

    return result == 0;
}

bool MdlPtxCodeGen::setOptionBinary(const char* name, const char* data, size_t size)
{
    if (mBackend->set_option_binary(name, data, size) != 0)
    {
        return false;
    }
    // limit functions for which PTX code is generated to the entry functions
    if (mBackend->set_option("visible_functions", "__closesthit__radiance") != 0)
    {
        STRELKA_ERROR("Setting PTX option visible_functions failed");
        return false;
    }
    return true;
}

} // namespace oka
