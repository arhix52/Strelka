//    Copyright (C) 2021 Pablo Delgado Krämer
//
//        This program is free software: you can redistribute it and/or modify
//    it under the terms of the GNU General Public License as published by
//    the Free Software Foundation, either version 3 of the License, or
//                                         (at your option) any later version.
//
//                                         This program is distributed in the hope that it will be useful,
//    but WITHOUT ANY WARRANTY; without even the implied warranty of
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
//    GNU General Public License for more details.
//
//    You should have received a copy of the GNU General Public License
//    along with this program. If not, see <https://www.gnu.org/licenses/>.

#include "mtlxMdlCodeGen.h"

#include <MaterialXCore/Definition.h>
#include <MaterialXCore/Document.h>
#include <MaterialXCore/Library.h>
#include <MaterialXCore/Material.h>
#include <MaterialXFormat/File.h>
#include <MaterialXFormat/Util.h>
#include <MaterialXGenMdl/MdlShaderGenerator.h>
#include <MaterialXGenShader/DefaultColorManagementSystem.h>
#include <MaterialXGenShader/GenContext.h>
#include <MaterialXGenShader/GenOptions.h>
#include <MaterialXGenShader/Library.h>
#include <MaterialXGenShader/Shader.h>
#include <MaterialXGenShader/Util.h>

#include <unordered_set>
#include <log.h>

namespace mx = MaterialX;

namespace oka
{

class MdlStringResolver;
using MdlStringResolverPtr = std::shared_ptr<MdlStringResolver>;


// Original source from https://github.com/NVIDIA/MDL-SDK/blob/190249748ddfe75b133b9da9028cc6272928c1b5/examples/mdl_sdk/dxr/mdl_d3d12/materialx/mdl_generator.cpp#L53
class MdlStringResolver : public mx::StringResolver
{
public:

    /// Create a new string resolver.
    static MdlStringResolverPtr create()
    {
        return MdlStringResolverPtr(new MdlStringResolver());
    }
    ~MdlStringResolver() = default;

    void initialize(mx::DocumentPtr document, mi::neuraylib::IMdl_configuration* config)
    {
        // remove duplicates and keep order by using a set
        auto less = [](const mx::FilePath& lhs, const mx::FilePath& rhs) { return lhs.asString() < rhs.asString(); };
        std::set<mx::FilePath, decltype(less)> mtlx_paths(less);
        m_mtlx_document_paths.clear();
        m_mdl_search_paths.clear();

        // use the source search paths as base
        mx::FilePath p = mx::FilePath(document->getSourceUri()).getParentPath().getNormalized();
        mtlx_paths.insert(p);
        m_mtlx_document_paths.append(p);

        for (auto sp : mx::getSourceSearchPath(document))
        {
            sp = sp.getNormalized();
            if(sp.exists() && mtlx_paths.insert(sp).second)
                m_mtlx_document_paths.append(sp);
        }

        // add all search paths known to MDL
        for (size_t i = 0, n = config->get_mdl_paths_length(); i < n; i++)
        {
            mi::base::Handle<const mi::IString> sp_istring(config->get_mdl_path(i));
            p = mx::FilePath(sp_istring->get_c_str()).getNormalized();
            if (p.exists() && mtlx_paths.insert(p).second)
                m_mtlx_document_paths.append(p);

            // keep a list of MDL search paths for resource resolution
            m_mdl_search_paths.append(p);
        }
    }

    std::string resolve(const std::string& str, const std::string& type) const override
    {
        mx::FilePath normalizedPath = mx::FilePath(str).getNormalized();

        // in case the path is absolute we need to find a proper search path to put the file in
        if (normalizedPath.isAbsolute())
        {
            // find the highest priority search path that is a prefix of the resource path
            for (const auto& sp : m_mdl_search_paths)
            {
                if (sp.size() > normalizedPath.size())
                    continue;

                bool isParent = true;
                for (size_t i = 0; i < sp.size(); ++i)
                {
                    if (sp[i] != normalizedPath[i])
                    {
                        isParent = false;
                        break;
                    }
                }

                if (!isParent)
                    continue;

                // found a search path that is a prefix of the resource
                std::string resource_path =
                    normalizedPath.asString(mx::FilePath::FormatPosix).substr(
                        sp.asString(mx::FilePath::FormatPosix).size());
                if (resource_path[0] != '/')
                    resource_path = "/" + resource_path;
                return resource_path;
            }
        }

        STRELKA_ERROR("MaterialX resource can not be accessed through an MDL search path. \n Dropping the resource from the Material. Resource Path: {} ", normalizedPath.asString());

        // drop the resource by returning the empty string.
        // alternatively, the resource could be copied into an MDL search path,
        // maybe even only temporary.
        return "";
    }

    // Get the MaterialX paths used to load the current document as well the current MDL search
    // paths in order to resolve resources by the MaterialX SDK.
    const mx::FileSearchPath& get_search_paths() const { return m_mtlx_document_paths; }

private:

    // List of paths from which MaterialX can locate resources.
    // This includes the document folder and the search paths used to load the document.
    mx::FileSearchPath m_mtlx_document_paths;

    // List of MDL search paths from which we can locate resources.
    // This is only a subset of the MaterialX document paths and needs to be extended by using the
    // `--mdl_path` option when starting the application if needed.
    mx::FileSearchPath m_mdl_search_paths;
};


MtlxMdlCodeGen::MtlxMdlCodeGen(const char* mtlxlibPath, MdlRuntime* mdlRuntime)
    : mMtlxlibPath(mtlxlibPath),
    mMdlRuntime(mdlRuntime)
{
    // Init shadergen.
    mShaderGen = mx::MdlShaderGenerator::create();
    std::string target = mShaderGen->getTarget();

    // MaterialX libs.
    mStdLib = mx::createDocument();
    mx::FilePathVec libFolders;
    mx::loadLibraries(libFolders, mMtlxlibPath, mStdLib);

    // Color management.
    mx::DefaultColorManagementSystemPtr colorSystem = mx::DefaultColorManagementSystem::create(target);
    colorSystem->loadLibrary(mStdLib);
    mShaderGen->setColorManagementSystem(colorSystem);

    // Unit management.
    mx::UnitSystemPtr unitSystem = mx::UnitSystem::create(target);
    unitSystem->loadLibrary(mStdLib);

    mx::UnitConverterRegistryPtr unitRegistry = mx::UnitConverterRegistry::create();
    mx::UnitTypeDefPtr distanceTypeDef = mStdLib->getUnitTypeDef("distance");
    unitRegistry->addUnitConverter(distanceTypeDef, mx::LinearUnitConverter::create(distanceTypeDef));
    mx::UnitTypeDefPtr angleTypeDef = mStdLib->getUnitTypeDef("angle");
    unitRegistry->addUnitConverter(angleTypeDef, mx::LinearUnitConverter::create(angleTypeDef));

    unitSystem->setUnitConverterRegistry(unitRegistry);
    mShaderGen->setUnitSystem(unitSystem);
}

mx::TypedElementPtr _FindSurfaceShaderElement(mx::DocumentPtr doc)
{
    // Find renderable element.
    std::vector<mx::TypedElementPtr> renderableElements;
    mx::findRenderableElements(doc, renderableElements);

    if (renderableElements.size() != 1)
    {
        return nullptr;
    }

    // Extract surface shader node.
    mx::TypedElementPtr renderableElement = renderableElements.at(0);
    mx::NodePtr node = renderableElement->asA<mx::Node>();

    if (node && node->getType() == mx::MATERIAL_TYPE_STRING)
    {
        auto shaderNodes = mx::getShaderNodes(node, mx::SURFACE_SHADER_TYPE_STRING);
        if (!shaderNodes.empty())
        {
            renderableElement = *shaderNodes.begin();
        }
    }

    mx::ElementPtr surfaceElement = doc->getDescendant(renderableElement->getNamePath());
    if (!surfaceElement)
    {
        return nullptr;
    }

    return surfaceElement->asA<mx::TypedElement>();
}

bool MtlxMdlCodeGen::translate(const char* mtlxSrc, std::string& mdlSrc, std::string& subIdentifier)
{
    // Don't cache the context because it is thread-local.
    mx::GenContext context(mShaderGen);
    context.registerSourceCodeSearchPath(mMtlxlibPath);

    mx::GenOptions& contextOptions = context.getOptions();
    contextOptions.targetDistanceUnit = "meter";

    mx::ShaderPtr shader = nullptr;
    try
    {
        mx::DocumentPtr doc = mx::createDocument();
        doc->importLibrary(mStdLib);
        mx::readFromXmlString(doc, mtlxSrc); // originally from string
        
        auto custom_resolver = MdlStringResolver::create();
        custom_resolver->initialize(doc, mMdlRuntime->getConfig().get());
        mx::flattenFilenames(doc, custom_resolver->get_search_paths(), custom_resolver);

        mx::TypedElementPtr element = _FindSurfaceShaderElement(doc);
        if (!element)
        {
            return false;
        }

        subIdentifier = element->getName();
        shader = mShaderGen->generate(subIdentifier, element, context);
    }
    catch (const std::exception& ex)
    {
        STRELKA_ERROR("Exception generating MDL code: {}", ex.what());
    }

    if (!shader)
    {
        return false;
    }

    mx::ShaderStage pixelStage = shader->getStage(mx::Stage::PIXEL);
    mdlSrc = pixelStage.getSourceCode();
    return true;
}
} // namespace oka
