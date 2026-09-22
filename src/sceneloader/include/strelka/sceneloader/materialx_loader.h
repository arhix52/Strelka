#pragma once

#include <strelka/material/openpbr/openpbr_params.h>
#include <strelka/scene/scene.h>

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace oka::mtlx
{

struct MaterialXMaterial
{
    /// The <surfacematerial> name, which is what a scene binds by.
    std::string name;
    OpenPBRParams params{};
    OpenPBRLayeredTextureParams layeredTexture{};
    /// Absolute paths, resolved against the document's own directory the way
    /// MaterialX resolves a filename input.
    std::array<std::string, MAX_OPENPBR_TEXTURES> texPaths;
    /// Per-slot override of the renderer's slot default, where the document
    /// stated a colorspace on the image it drives the slot from.
    std::array<TexColorSpace, MAX_OPENPBR_TEXTURES> texColorSpace{};
    /// Node categories found upstream of an input that could not be resolved.
    /// Empty when the document was fully understood.
    std::vector<std::string> unsupported;
};

struct MaterialXAssignment
{
    std::string geom;
    std::string material;
};

struct MaterialXDocumentData
{
    std::vector<MaterialXMaterial> materials;
    std::vector<MaterialXAssignment> assignments;
};

/// Reads every surface material and look assignment in `path`. Returns empty
/// and logs on failure.
MaterialXDocumentData loadMaterialXDocument(const std::string& path);

int applyMaterialXDocument(Scene& scene, const std::string& path);

} // namespace oka::mtlx
