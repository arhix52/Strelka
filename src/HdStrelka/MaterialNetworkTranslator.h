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

#pragma once

#include <pxr/usd/sdf/path.h>

#include <string>
#include <MaterialXCore/Document.h>

#include <memory>


PXR_NAMESPACE_OPEN_SCOPE

struct HdMaterialNetwork2;

class MaterialNetworkTranslator
{
public:
    MaterialNetworkTranslator(const std::string& mtlxLibPath);

    std::string ParseNetwork(const SdfPath& id, const HdMaterialNetwork2& network) const;
    static bool ParseMdlNetwork(const HdMaterialNetwork2& network, std::string& fileUri, std::string& subIdentifier);

private:
    MaterialX::DocumentPtr CreateMaterialXDocumentFromNetwork(const SdfPath& id, const HdMaterialNetwork2& network) const;
    void patchMaterialNetwork(HdMaterialNetwork2& network) const;

private:
    MaterialX::DocumentPtr m_nodeLib;
};

PXR_NAMESPACE_CLOSE_SCOPE
