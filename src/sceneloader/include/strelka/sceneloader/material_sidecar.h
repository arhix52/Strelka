#pragma once

// ============================================================================
// material_sidecar.h -- <stem>_openpbr.json, an OpenPBR material per glTF one
// ============================================================================
//
// glTF cannot express OpenPBR. Its material extensions stop at a clearcoat with
// no darkening and a sheen that is a different lobe, and there is no ratified
// MaterialX extension to point at either. This codebase has already answered
// that question twice -- analytic lights live in <stem>_light.json and hair in
// <stem>_curves.bin -- so this is the third instance of the same pattern rather
// than a new idea.
//
// A material named here is authored, not translated: the whole OpenPBR parameter
// set is written out and used as given. That is what makes coat_darkening,
// dispersion, a fuzz layer and a per-channel subsurface radius reachable at all,
// none of which survive a trip through glTF.
//
//   <stem>_openpbr.json:
//
//   {
//     "version": 1,
//     "materials": [
//       { "gltfMaterial": "Ceramic",
//         "openpbr": { "base_color": [0.9, 0.9, 0.88],
//                      "specular_roughness": 0.08,
//                      "coat_weight": 1.0,
//                      "coat_darkening": 1.0 } }
//     ]
//   }
//
// Anything left out keeps its OpenPBR 1.1.1 default, so a file states only what
// it changes. Unknown keys are a warning rather than silence: in a format where
// a misspelt key simply does nothing, silence is the failure mode that costs an
// afternoon.
//
// Materials not named here are untouched and keep shading through the glTF
// model, so a sidecar can convert one object in a scene.

#include <strelka/material/openpbr/openpbr_params.h>
#include <strelka/scene/scene.h>

#include <log.h>

#include "nlohmann/json.hpp"

#include <filesystem>
#include <fstream>
#include <string>
#include <array>
#include <unordered_map>
#include <unordered_set>

namespace oka::materialsidecar
{

inline bool readColor(const nlohmann::json& j, OpenPBRColor& out)
{
    if (!j.is_array() || j.size() < 3)
    {
        return false;
    }
    out = OpenPBRColor{ j[0].get<float>(), j[1].get<float>(), j[2].get<float>() };
    return true;
}

/// Fills `p` from one "openpbr" object. Returns the number of keys it did not
/// recognise, having logged each.
inline int parseOpenPBR(const nlohmann::json& j, OpenPBRParams& p)
{
    int unknown = 0;
    std::unordered_set<std::string> seen;

    auto scalar = [&](const char* key, float& dst) {
        seen.insert(key);
        if (j.contains(key))
            dst = j[key].get<float>();
    };
    auto color = [&](const char* key, OpenPBRColor& dst) {
        seen.insert(key);
        if (j.contains(key) && !readColor(j[key], dst))
            STRELKA_WARNING("materials sidecar: '{}' must be an array of three numbers", key);
    };
    auto flag = [&](const char* key, unsigned int& dst) {
        seen.insert(key);
        if (j.contains(key))
            dst = j[key].get<bool>() ? 1u : 0u;
    };

    scalar("base_weight", p.base_weight);
    color("base_color", p.base_color);
    scalar("base_diffuse_roughness", p.base_diffuse_roughness);
    scalar("base_metalness", p.base_metalness);

    scalar("subsurface_weight", p.subsurface_weight);
    color("subsurface_color", p.subsurface_color);
    scalar("subsurface_radius", p.subsurface_radius);
    color("subsurface_radius_scale", p.subsurface_radius_scale);
    scalar("subsurface_scatter_anisotropy", p.subsurface_scatter_anisotropy);

    scalar("specular_weight", p.specular_weight);
    color("specular_color", p.specular_color);
    scalar("specular_roughness", p.specular_roughness);
    scalar("specular_roughness_anisotropy", p.specular_roughness_anisotropy);
    scalar("specular_ior", p.specular_ior);

    scalar("coat_weight", p.coat_weight);
    color("coat_color", p.coat_color);
    scalar("coat_roughness", p.coat_roughness);
    scalar("coat_roughness_anisotropy", p.coat_roughness_anisotropy);
    scalar("coat_ior", p.coat_ior);
    scalar("coat_darkening", p.coat_darkening);

    scalar("fuzz_weight", p.fuzz_weight);
    color("fuzz_color", p.fuzz_color);
    scalar("fuzz_roughness", p.fuzz_roughness);

    scalar("transmission_weight", p.transmission_weight);
    color("transmission_color", p.transmission_color);
    scalar("transmission_depth", p.transmission_depth);
    color("transmission_scatter", p.transmission_scatter);
    scalar("transmission_scatter_anisotropy", p.transmission_scatter_anisotropy);
    scalar("transmission_dispersion_scale", p.transmission_dispersion_scale);
    scalar("transmission_dispersion_abbe_number", p.transmission_dispersion_abbe_number);

    scalar("thin_film_weight", p.thin_film_weight);
    scalar("thin_film_thickness", p.thin_film_thickness);
    scalar("thin_film_ior", p.thin_film_ior);

    scalar("emission_luminance", p.emission_luminance);
    color("emission_color", p.emission_color);

    scalar("geometry_opacity", p.geometry_opacity);
    flag("geometry_thin_walled", p.geometry_thin_walled);

    // The spec keeps rotations as angles; the runtime keeps their cosine and
    // sine so a filtered value cannot wrap. Authors write the angle.
    seen.insert("specular_anisotropy_rotation");
    if (j.contains("specular_anisotropy_rotation"))
    {
        const float a = j["specular_anisotropy_rotation"].get<float>();
        p.specular_anisotropy_rotation_cos = std::cos(a);
        p.specular_anisotropy_rotation_sin = std::sin(a);
    }
    seen.insert("coat_anisotropy_rotation");
    if (j.contains("coat_anisotropy_rotation"))
    {
        const float a = j["coat_anisotropy_rotation"].get<float>();
        p.coat_anisotropy_rotation_cos = std::cos(a);
        p.coat_anisotropy_rotation_sin = std::sin(a);
    }

    for (const auto& item : j.items())
    {
        if (seen.find(item.key()) == seen.end())
        {
            STRELKA_WARNING("materials sidecar: unknown OpenPBR parameter '{}', ignored", item.key());
            ++unknown;
        }
    }
    return unknown;
}

/// Slot name as written in the file -> OpenPBRTextureSlot, or -1.
///
/// Spelled out rather than derived from the parameter list because the two are
/// not the same set: only sixteen inputs have a slot, and a map named for one of
/// the others has to be reported rather than quietly dropped.
inline int textureSlotFromName(const std::string& name)
{
    static const std::unordered_map<std::string, int> kSlots = {
        { "base_color", OPENPBR_TEX_BASE_COLOR },
        { "base_metalness", OPENPBR_TEX_BASE_METALNESS },
        { "specular_roughness", OPENPBR_TEX_SPECULAR_ROUGHNESS },
        { "specular_color", OPENPBR_TEX_SPECULAR_COLOR },
        { "specular_roughness_anisotropy", OPENPBR_TEX_SPECULAR_ANISOTROPY },
        { "coat_weight", OPENPBR_TEX_COAT_WEIGHT },
        { "coat_roughness", OPENPBR_TEX_COAT_ROUGHNESS },
        { "coat_color", OPENPBR_TEX_COAT_COLOR },
        { "fuzz_weight", OPENPBR_TEX_FUZZ_WEIGHT },
        { "fuzz_roughness", OPENPBR_TEX_FUZZ_ROUGHNESS },
        { "emission_color", OPENPBR_TEX_EMISSION_COLOR },
        { "transmission_color", OPENPBR_TEX_TRANSMISSION_COLOR },
        { "subsurface_color", OPENPBR_TEX_SUBSURFACE_COLOR },
        { "geometry_normal", OPENPBR_TEX_GEOMETRY_NORMAL },
        { "geometry_coat_normal", OPENPBR_TEX_GEOMETRY_COAT_NORMAL },
        { "geometry_opacity", OPENPBR_TEX_GEOMETRY_OPACITY },
        { "subsurface_weight", OPENPBR_TEX_SUBSURFACE_WEIGHT },
        { "subsurface_radius_scale", OPENPBR_TEX_SUBSURFACE_RADIUS },
        { "fuzz_color", OPENPBR_TEX_FUZZ_COLOR },
    };
    const auto it = kSlots.find(name);
    return it == kSlots.end() ? -1 : it->second;
}

/// Fills `paths` from one "textures" object. Returns the number of unrecognised
/// slot names, having logged each.
inline int parseTextures(const nlohmann::json& j, std::array<std::string, MAX_OPENPBR_TEXTURES>& paths)
{
    int unknown = 0;
    for (const auto& item : j.items())
    {
        const int slot = textureSlotFromName(item.key());
        if (slot < 0)
        {
            STRELKA_WARNING("materials sidecar: '{}' has no texture slot, map ignored", item.key());
            ++unknown;
            continue;
        }
        // Relative to the scene, exactly like a glTF image URI: the renderer
        // joins it with resource/searchPath, and the sidecar sits beside the
        // scene it describes.
        paths[(size_t)slot] = item.value().get<std::string>();
    }
    return unknown;
}

/// Applies `path` to `scene`. Returns the number of materials it changed.
inline int loadMaterialsJson(Scene& scene, const std::string& path)
{
    std::ifstream file(path);
    if (!file.is_open())
    {
        return 0;
    }

    nlohmann::json doc;
    try
    {
        file >> doc;
    }
    catch (const std::exception& e)
    {
        STRELKA_ERROR("materials sidecar {}: {}", path, e.what());
        return 0;
    }

    // Refuse anything that is not this format rather than throw through the
    // scene loader: a file that merely shares a name is a mistake to report, not
    // a reason for the renderer to abort before its first frame.
    if (!doc.is_object())
    {
        STRELKA_WARNING("materials sidecar {}: not an object, ignored", path);
        return 0;
    }

    const int version = doc.value("version", 1);
    if (version != 1)
    {
        STRELKA_WARNING("materials sidecar {}: version {} is newer than this build understands", path, version);
    }
    if (!doc.contains("materials") || !doc["materials"].is_array())
    {
        STRELKA_WARNING("materials sidecar {}: no \"materials\" array", path);
        return 0;
    }

    std::vector<Scene::MaterialDescription>& descs = scene.getMaterials();
    int applied = 0;
    for (const auto& entry : doc["materials"])
    {
        const std::string name = entry.value("gltfMaterial", std::string());
        if (name.empty())
        {
            STRELKA_WARNING("materials sidecar {}: an entry has no \"gltfMaterial\"", path);
            continue;
        }
        if (!entry.contains("openpbr"))
        {
            STRELKA_WARNING("materials sidecar {}: '{}' has no \"openpbr\" block", path, name);
            continue;
        }

        bool matched = false;
        for (Scene::MaterialDescription& desc : descs)
        {
            if (desc.name != name)
            {
                continue;
            }
            // Start from the spec defaults, not from whatever the glTF material
            // happened to be: an authored material is a statement of the whole
            // surface, and inheriting half of a different model would make the
            // result depend on what the exporter wrote.
            desc.openpbr = openpbr_make_default_params();
            parseOpenPBR(entry["openpbr"], desc.openpbr);
            desc.openpbrTexPaths = {};
            if (entry.contains("textures"))
            {
                parseTextures(entry["textures"], desc.openpbrTexPaths);
            }
            desc.params.material_type = MATERIAL_TYPE_OPENPBR;
            matched = true;
            ++applied;
        }
        if (!matched)
        {
            STRELKA_WARNING("materials sidecar {}: no glTF material named '{}'", path, name);
        }
    }
    STRELKA_INFO("materials sidecar {}: {} material(s) authored as OpenPBR", path, applied);
    return applied;
}

/// <stem>_openpbr.json beside the scene, or empty when there is none.
///
/// Not "_materials.json": that name is taken. The V-Ray converter already
/// writes one beside every scene it produces, holding an array of
/// {material, kind, extra} records -- iso_bathroom has one -- and reading it as
/// this format threw an uncaught nlohmann type_error before the first frame.
/// The name also says less than it should; this file is specifically OpenPBR.
inline std::string findMaterialSidecar(const std::string& sceneFileNoExt)
{
    const std::string candidate = sceneFileNoExt + "_openpbr.json";
    return std::filesystem::exists(candidate) ? candidate : std::string();
}

} // namespace oka::materialsidecar
