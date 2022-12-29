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

#include "MdlParserPlugin.h"

#include <pxr/base/tf/staticTokens.h>

#include <pxr/usd/sdr/shaderNode.h>
#include <pxr/usd/ar/resolver.h>
#include "pxr/usd/ar/resolvedPath.h"
#include "pxr/usd/ar/asset.h"
#include <pxr/usd/ar/ar.h>

//#include "Tokens.h"

PXR_NAMESPACE_OPEN_SCOPE

NDR_REGISTER_PARSER_PLUGIN(HdStrelkaMdlParserPlugin);

// clang-format off
TF_DEFINE_PRIVATE_TOKENS(_tokens,
    (mdl)
    (subIdentifier));
// clang-format on

NdrNodeUniquePtr HdStrelkaMdlParserPlugin::Parse(const NdrNodeDiscoveryResult& discoveryResult)
{
    NdrTokenMap metadata = discoveryResult.metadata;
    metadata[_tokens->subIdentifier] = discoveryResult.subIdentifier;

    return std::make_unique<SdrShaderNode>(discoveryResult.identifier, discoveryResult.version, discoveryResult.name,
                                           discoveryResult.family, _tokens->mdl, discoveryResult.sourceType,
                                           discoveryResult.uri, discoveryResult.resolvedUri, NdrPropertyUniquePtrVec{},
                                           metadata);
}

const NdrTokenVec& HdStrelkaMdlParserPlugin::GetDiscoveryTypes() const
{
    static NdrTokenVec s_discoveryTypes{ _tokens->mdl };
    return s_discoveryTypes;
}

const TfToken& HdStrelkaMdlParserPlugin::GetSourceType() const
{
    return _tokens->mdl;
}

PXR_NAMESPACE_CLOSE_SCOPE
