#pragma once

// ============================================================================
// materialx_loader.h -- a .mtlx document read as OpenPBR materials
// ============================================================================
//
// MaterialX enters this renderer as a *front end*, not as a shader generator.
// MaterialXGenMsl exists and there is no CUDA equivalent, so generating code
// from a document would make MaterialX a Metal-only feature; instead the
// document is parsed on the host and its surface shaders are resolved into the
// OpenPBR parameter block both backends already understand.
//
// What that buys, and what it costs:
//
//   + one implementation for Metal and OptiX, and the same one the unit tests
//     exercise on the CPU;
//   - only inputs that resolve to a constant or to a single image are honoured.
//     A procedural graph has nowhere to go without baking, and baking would drag
//     MaterialXRender and a GL or Metal context into the scene loader.
//
// Supported upstream of a shader input, which is exactly what the shipped
// example materials and the Open Chess Set use:
//
//   value="..."                    -> a constant
//   <image file="..."/>            -> a texture slot
//   <tiledimage file="..."/>       -> a texture slot
//   <normalmap in="<image>"/>      -> the normal slot
//
// Anything else is reported by node category and the input keeps its OpenPBR
// default. Silence would be worse: a material that quietly lost its base colour
// still renders, just wrongly.
//
// Both open_pbr_surface and standard_surface are accepted. The second is
// mapped -- OpenPBR unifies Autodesk Standard Surface with Adobe Standard
// Material, so most of it is a rename, and the places where it is not are
// commented at the mapping.

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

/// One <materialassign> out of a <look>: which material goes on which geometry.
///
/// Needed because binding by material name is not how a MaterialX scene is
/// usually put together. The Open Chess Set is the case in point: its glTF
/// carries two placeholder materials called "Default OBJ" for fifteen pieces,
/// and the document assigns by geometry name instead.
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

/// Reads `path` and applies it to `scene`, matching a MaterialX material to a
/// glTF one by name. `M_Bishop_B` also matches a glTF material named
/// `Bishop_B`: exporters routinely prefix, and requiring an exact match would
/// make the common case fail silently.
///
/// Returns the number of scene materials changed.
int applyMaterialXDocument(Scene& scene, const std::string& path);

} // namespace oka::mtlx
