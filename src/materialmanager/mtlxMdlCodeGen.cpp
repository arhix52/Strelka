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
MtlxMdlCodeGen::MtlxMdlCodeGen(const char* mtlxlibPath)
    : mMtlxlibPath(mtlxlibPath)
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

        printf("mtlx code: %s\n", mtlxSrc);

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
