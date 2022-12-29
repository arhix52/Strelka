#pragma once

#include "mdlLogger.h"
#include "mdlRuntime.h"

#include <MaterialXCore/Document.h>
#include <MaterialXFormat/File.h>
#include <MaterialXGenShader/ShaderGenerator.h>
#include <mi/mdl_sdk.h>

namespace oka
{
class MdlPtxCodeGen
{
public:
    explicit MdlPtxCodeGen(){};
    bool init(MdlRuntime& runtime);

    struct InternalMaterialInfo
    {
        mi::Size argument_block_index;
    };

    bool setOptionBinary(const char* name, const char* data, size_t size);

    mi::base::Handle<const mi::neuraylib::ITarget_code> translate(
        const mi::neuraylib::ICompiled_material* material,
        std::string& ptxSrc,
        InternalMaterialInfo& internalsInfo);

    mi::base::Handle<const mi::neuraylib::ITarget_code> translate(
        const std::vector<const mi::neuraylib::ICompiled_material*>& materials,
        std::string& ptxSrc,
        std::vector<InternalMaterialInfo>& internalsInfo);


private:
    bool appendMaterialToLinkUnit(uint32_t idx,
                                  const mi::neuraylib::ICompiled_material* compiledMaterial,
                                  mi::neuraylib::ILink_unit* linkUnit,
                                  mi::Size& argBlockIndex);

    std::unique_ptr<MdlNeurayLoader> mLoader;

    mi::base::Handle<MdlLogger> mLogger;
    mi::base::Handle<mi::neuraylib::IMdl_backend> mBackend;
    mi::base::Handle<mi::neuraylib::IDatabase> mDatabase;
    mi::base::Handle<mi::neuraylib::ITransaction> mTransaction;
    mi::base::Handle<mi::neuraylib::IMdl_execution_context> mContext;
};
} // namespace oka
