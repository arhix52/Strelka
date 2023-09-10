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

#include "MaterialNetworkTranslator.h"

#include <pxr/imaging/hd/material.h>
#include <pxr/usd/sdr/registry.h>
#include <pxr/imaging/hdMtlx/hdMtlx.h>

#include <pxr/imaging/hd/tokens.h>

#include <MaterialXCore/Document.h>
#include <MaterialXCore/Library.h>
#include <MaterialXCore/Material.h>
#include <MaterialXCore/Definition.h>
#include <MaterialXFormat/File.h>
#include <MaterialXFormat/Util.h>

#include "Tokens.h"

namespace mx = MaterialX;

PXR_NAMESPACE_OPEN_SCOPE

// clang-format off
TF_DEFINE_PRIVATE_TOKENS(
  _tokens,
  // USD node types
  (UsdPreviewSurface)
  (UsdUVTexture)
  (UsdTransform2d)
  (UsdPrimvarReader_float)
  (UsdPrimvarReader_float2)
  (UsdPrimvarReader_float3)
  (UsdPrimvarReader_float4)
  (UsdPrimvarReader_int)
  (UsdPrimvarReader_string)
  (UsdPrimvarReader_normal)
  (UsdPrimvarReader_point)
  (UsdPrimvarReader_vector)
  (UsdPrimvarReader_matrix)
  // MaterialX USD node type equivalents
  (ND_UsdPreviewSurface_surfaceshader)
  (ND_UsdUVTexture)
  (ND_UsdPrimvarReader_integer)
  (ND_UsdPrimvarReader_boolean)
  (ND_UsdPrimvarReader_string)
  (ND_UsdPrimvarReader_float)
  (ND_UsdPrimvarReader_vector2)
  (ND_UsdPrimvarReader_vector3)
  (ND_UsdPrimvarReader_vector4)
  (ND_UsdTransform2d)
  (ND_UsdPrimvarReader_matrix44)
  (mdl)
  (subIdentifier)
  (ND_convert_color3_vector3)
  (diffuse_color_constant)
  (normal)
  (rgb)
  (in)
  (out)
);
// clang-format on

bool _ConvertNodesToMaterialXNodes(const HdMaterialNetwork2& network, HdMaterialNetwork2& mtlxNetwork)
{
    mtlxNetwork = network;

    for (auto nodeIt = mtlxNetwork.nodes.begin(); nodeIt != mtlxNetwork.nodes.end(); nodeIt++)
    {
        TfToken& nodeTypeId = nodeIt->second.nodeTypeId;

        SdrRegistry& sdrRegistry = SdrRegistry::GetInstance();
        if (sdrRegistry.GetShaderNodeByIdentifierAndType(nodeTypeId, HdStrelkaDiscoveryTypes->mtlx))
        {
            continue;
        }

        if (nodeTypeId == _tokens->UsdPreviewSurface)
        {
            nodeTypeId = _tokens->ND_UsdPreviewSurface_surfaceshader;
        }
        else if (nodeTypeId == _tokens->UsdUVTexture)
        {
            nodeTypeId = _tokens->ND_UsdUVTexture;
        }
        else if (nodeTypeId == _tokens->UsdTransform2d)
        {
            nodeTypeId = _tokens->ND_UsdTransform2d;
        }
        else if (nodeTypeId == _tokens->UsdPrimvarReader_float)
        {
            nodeTypeId = _tokens->ND_UsdPrimvarReader_float;
        }
        else if (nodeTypeId == _tokens->UsdPrimvarReader_float2)
        {
            nodeTypeId = _tokens->ND_UsdPrimvarReader_vector2;
        }
        else if (nodeTypeId == _tokens->UsdPrimvarReader_float3)
        {
            nodeTypeId = _tokens->ND_UsdPrimvarReader_vector3;
        }
        else if (nodeTypeId == _tokens->UsdPrimvarReader_float4)
        {
            nodeTypeId = _tokens->ND_UsdPrimvarReader_vector4;
        }
        else if (nodeTypeId == _tokens->UsdPrimvarReader_int)
        {
            nodeTypeId = _tokens->ND_UsdPrimvarReader_integer;
        }
        else if (nodeTypeId == _tokens->UsdPrimvarReader_string)
        {
            nodeTypeId = _tokens->ND_UsdPrimvarReader_string;
        }
        else if (nodeTypeId == _tokens->UsdPrimvarReader_normal)
        {
            nodeTypeId = _tokens->ND_UsdPrimvarReader_vector3;
        }
        else if (nodeTypeId == _tokens->UsdPrimvarReader_point)
        {
            nodeTypeId = _tokens->ND_UsdPrimvarReader_vector3;
        }
        else if (nodeTypeId == _tokens->UsdPrimvarReader_vector)
        {
            nodeTypeId = _tokens->ND_UsdPrimvarReader_vector3;
        }
        else if (nodeTypeId == _tokens->UsdPrimvarReader_matrix)
        {
            nodeTypeId = _tokens->ND_UsdPrimvarReader_matrix44;
        }
        else
        {
            TF_WARN("Unable to translate material node of type %s to MaterialX counterpart", nodeTypeId.GetText());
            return false;
        }
    }

    return true;
}

bool _GetMaterialNetworkSurfaceTerminal(const HdMaterialNetwork2& network2,
                                        HdMaterialNode2& surfaceTerminal,
                                        SdfPath& terminalPath)
{
    const auto& connectionIt = network2.terminals.find(HdMaterialTerminalTokens->surface);

    if (connectionIt == network2.terminals.end())
    {
        return false;
    }

    const HdMaterialConnection2& connection = connectionIt->second;

    terminalPath = connection.upstreamNode;

    const auto& nodeIt = network2.nodes.find(terminalPath);

    if (nodeIt == network2.nodes.end())
    {
        return false;
    }

    surfaceTerminal = nodeIt->second;

    return true;
}

MaterialNetworkTranslator::MaterialNetworkTranslator(const std::string& mtlxLibPath)
{
    m_nodeLib = mx::createDocument();

    mx::FilePathVec libFolders; // All directories if left empty.
    mx::FileSearchPath folderSearchPath(mtlxLibPath);
    mx::loadLibraries(libFolders, folderSearchPath, m_nodeLib);
}

std::string MaterialNetworkTranslator::ParseNetwork(const SdfPath& id, const HdMaterialNetwork2& network) const
{
    HdMaterialNetwork2 mtlxNetwork;
    if (!_ConvertNodesToMaterialXNodes(network, mtlxNetwork))
    {
        return nullptr;
    }

    patchMaterialNetwork(mtlxNetwork);

    mx::DocumentPtr doc = CreateMaterialXDocumentFromNetwork(id, mtlxNetwork);
    if (!doc)
    {
        return nullptr;
    }

    mx::string docStr = mx::writeToXmlString(doc);

    return std::string(docStr.c_str());
}

bool MaterialNetworkTranslator::ParseMdlNetwork(const SdfPath& id,
                                                const HdMaterialNetwork2& network,
                                                std::string& fileUri,
                                                std::string& subIdentifier) const
{
    if (network.nodes.size() == 1)
    {
        const HdMaterialNode2& node = network.nodes.begin()->second;

        SdrRegistry& sdrRegistry = SdrRegistry::GetInstance();
        SdrShaderNodeConstPtr sdrNode = sdrRegistry.GetShaderNodeByIdentifier(node.nodeTypeId);

        if (!sdrNode || sdrNode->GetContext() != _tokens->mdl)
        {
            return false;
        }

        const NdrTokenMap& metadata = sdrNode->GetMetadata();
        const auto& subIdentifierIt = metadata.find(_tokens->subIdentifier);
        TF_DEV_AXIOM(subIdentifierIt != metadata.end());

        subIdentifier = (*subIdentifierIt).second;
        fileUri = sdrNode->GetResolvedImplementationURI();

        return true;
    }
    else
    {
        TF_RUNTIME_ERROR("Unsupported multi-node MDL material!");
        return false;
    }
}

mx::DocumentPtr MaterialNetworkTranslator::CreateMaterialXDocumentFromNetwork(const SdfPath& id,
                                                                              const HdMaterialNetwork2& network) const
{
    HdMaterialNode2 surfaceTerminal;
    SdfPath terminalPath;
    if (!_GetMaterialNetworkSurfaceTerminal(network, surfaceTerminal, terminalPath))
    {
        TF_WARN("Unable to find surface terminal for material network");
        return nullptr;
    }

    HdMtlxTexturePrimvarData mxHdData;

    return HdMtlxCreateMtlxDocumentFromHdNetwork(network, surfaceTerminal, terminalPath, id, m_nodeLib, &mxHdData);
}

void MaterialNetworkTranslator::patchMaterialNetwork(HdMaterialNetwork2& network) const
{
    for (auto& pathNodePair : network.nodes)
    {
        HdMaterialNode2& node = pathNodePair.second;
        if (node.nodeTypeId != _tokens->ND_UsdPreviewSurface_surfaceshader)
        {
            continue;
        }
        auto& inputs = node.inputConnections;

        const auto patchColor3Vector3InputConnection = [&inputs, &network](TfToken inputName) {
            auto inputIt = inputs.find(inputName);
            if (inputIt == inputs.end())
            {
                return;
            }

            auto& connections = inputIt->second;
            for (HdMaterialConnection2& connection : connections)
            {
                if (connection.upstreamOutputName != _tokens->rgb)
                {
                    continue;
                }

                SdfPath upstreamNodePath = connection.upstreamNode;

                SdfPath convertNodePath = upstreamNodePath;
                for (int i = 0; network.nodes.count(convertNodePath) > 0; i++)
                {
                    std::string convertNodeName = "convert" + std::to_string(i);
                    convertNodePath = upstreamNodePath.AppendElementString(convertNodeName);
                }

                HdMaterialNode2 convertNode;
                convertNode.nodeTypeId = _tokens->ND_convert_color3_vector3;
                convertNode.inputConnections[_tokens->in] = { { upstreamNodePath, _tokens->rgb } };
                network.nodes[convertNodePath] = convertNode;

                connection.upstreamNode = convertNodePath;
                connection.upstreamOutputName = _tokens->out;
            }
        };

        patchColor3Vector3InputConnection(_tokens->normal);
    }
}

PXR_NAMESPACE_CLOSE_SCOPE
