#pragma once

#include <strelka/scene/scene.h>
#include <strelka/scene/light_desc.h>
#include <strelka/sceneloader/iesloader.h>

#include "nlohmann/json.hpp"

#include <cmath>
#include <filesystem>
#include <string>
#include <numbers>


namespace oka::lightjson
{

inline Scene::UniformLightDesc parseDesc(const nlohmann::json& light, const std::string& searchDir = {})
{
    Scene::UniformLightDesc desc{};
    desc.useXform = false;
    desc.enabled = light.value("enabled", true);
    desc.visibleToCamera = light.value("visibleToCamera", true);
    desc.responsive = light.value("responsive", false);
    desc.name = light.value("name", "");
    desc.type = lightTypeFromName(light.value("type", "rect"));
    desc.intensityUnit = lightUnitFromName(light.value("unit", "radiance"));

    if (light.contains("orientation"))
    {
        const auto& o = light["orientation"];
        desc.orientation = glm::float3(o[0], o[1], o[2]);
    }
    if (light.contains("color"))
    {
        const auto& c = light["color"];
        desc.color = glm::float3(c[0], c[1], c[2]);
    }
    if (light.contains("intensity"))
        desc.intensity = light["intensity"].get<float>();
    if (light.contains("range"))
        desc.range = light["range"].get<float>();

    if (desc.type != LIGHT_TYPE_DISTANT && desc.type != LIGHT_TYPE_DOME && light.contains("position"))
    {
        const auto& p = light["position"];
        desc.position = glm::float3(p[0], p[1], p[2]);
    }

    switch (desc.type)
    {
    case LIGHT_TYPE_DISTANT:
        // JSON stores full angular diameter in degrees; GPU wants half-angle rad.
        desc.halfAngle = light.value("halfAngle", 0.53f) * 0.5f * (std::numbers::pi_v<float> / 180.0f);
        if (!light.contains("unit"))
            desc.intensityUnit = LIGHT_UNIT_RADIANCE;
        break;
    case LIGHT_TYPE_DOME:
        if (!light.contains("unit"))
            desc.intensityUnit = LIGHT_UNIT_RADIANCE;
        break;
    case LIGHT_TYPE_SPHERE:
        desc.radius = light.value("radius", 0.1f);
        break;
    case LIGHT_TYPE_DISC:
        desc.radius = light.value("radius", 0.5f);
        break;
    case LIGHT_TYPE_POINT:
        desc.radius = light.value("radius", 0.0f);
        if (!light.contains("unit"))
            desc.intensityUnit = LIGHT_UNIT_INTENSITY;
        break;
    case LIGHT_TYPE_SPOT:
        desc.radius = light.value("radius", 0.0f);
        // Degrees in JSON, radians on the desc — matches Blender / KHR UX.
        desc.innerConeAngle = light.value("innerConeAngle", 0.0f) * (std::numbers::pi_v<float> / 180.0f);
        desc.outerConeAngle = light.value("outerConeAngle", 45.0f) * (std::numbers::pi_v<float> / 180.0f);
        if (!light.contains("unit"))
            desc.intensityUnit = LIGHT_UNIT_INTENSITY;
        break;
    case LIGHT_TYPE_PROJECTOR:
        desc.radius = light.value("radius", 0.0f);
        // The *full* horizontal field of view in degrees, the way a projector or
        // a camera is specified, halved into the outer-cone field the GPU light
        // already has. A spot's "outerConeAngle" is a half angle in the same
        // file, which reads like an inconsistency and is not one: nobody
        // describes a beamer by half its throw angle.
        desc.outerConeAngle = light.value("fov", 45.0f) * 0.5f * (std::numbers::pi_v<float> / 180.0f);
        desc.projectorAspect = light.value("aspect", 16.0f / 9.0f);
        desc.projectorEdgeSoftness = light.value("edgeSoftness", 0.0f);
        if (light.contains("image"))
            desc.projectorImagePath = light["image"].get<std::string>();
        if (!light.contains("unit"))
            desc.intensityUnit = LIGHT_UNIT_INTENSITY;
        break;
    case LIGHT_TYPE_RECT:
    default:
        desc.type = LIGHT_TYPE_RECT;
        desc.width = light.value("width", 1.0f);
        desc.height = light.value("height", 1.0f);
        break;
    }

    if (light.contains("ies") && !searchDir.empty())
    {
        desc.iesPath = light["ies"].get<std::string>();
    }
    return desc;
}

inline nlohmann::json toJson(const Scene::UniformLightDesc& desc)
{
    nlohmann::json light;
    light["type"] = lightTypeName(desc.type);
    light["enabled"] = desc.enabled;
    if (!desc.visibleToCamera)
        light["visibleToCamera"] = false;
    if (desc.responsive)
        light["responsive"] = true;
    if (!desc.name.empty())
        light["name"] = desc.name;
    light["color"] = { desc.color.x, desc.color.y, desc.color.z };
    light["intensity"] = desc.intensity;
    light["unit"] = lightUnitName(desc.intensityUnit);
    light["orientation"] = { desc.orientation.x, desc.orientation.y, desc.orientation.z };

    if (desc.type == LIGHT_TYPE_DISTANT)
    {
        light["halfAngle"] = desc.halfAngle * 2.0f * (180.0f / std::numbers::pi_v<float>);
    }
    else if (desc.type != LIGHT_TYPE_DOME)
    {
        light["position"] = { desc.position.x, desc.position.y, desc.position.z };
        if (desc.type == LIGHT_TYPE_RECT)
        {
            light["width"] = desc.width;
            light["height"] = desc.height;
        }
        else if (desc.type == LIGHT_TYPE_DISC || desc.type == LIGHT_TYPE_SPHERE || desc.type == LIGHT_TYPE_POINT ||
                 desc.type == LIGHT_TYPE_SPOT || desc.type == LIGHT_TYPE_PROJECTOR)
        {
            light["radius"] = desc.radius;
        }
        if (desc.type == LIGHT_TYPE_SPOT)
        {
            light["innerConeAngle"] = desc.innerConeAngle * (180.0f / std::numbers::pi_v<float>);
            light["outerConeAngle"] = desc.outerConeAngle * (180.0f / std::numbers::pi_v<float>);
        }
        if (desc.type == LIGHT_TYPE_PROJECTOR)
        {
            light["fov"] = desc.outerConeAngle * 2.0f * (180.0f / std::numbers::pi_v<float>);
            light["aspect"] = desc.projectorAspect;
            if (desc.projectorEdgeSoftness > 0.0f)
                light["edgeSoftness"] = desc.projectorEdgeSoftness;
            if (!desc.projectorImagePath.empty())
                light["image"] = desc.projectorImagePath;
        }
        if (desc.range > 0.0f)
            light["range"] = desc.range;
        if (!desc.iesPath.empty())
            light["ies"] = desc.iesPath;
    }
    return light;
}

/// Turn a projector's image path into an index into the scene's image table.
///
/// The mirror of resolveIes() below, and split from parseDesc() for the same
/// reason: parsing is a pure function of the JSON, while registering a resource
/// needs the Scene. The path is made absolute here so that a sidecar can name
/// the file next to itself and the renderer, which resolves everything else
/// against `resource/searchPath`, still finds it.
inline void resolveProjectorImage(Scene& scene, Scene::UniformLightDesc& desc, const std::string& searchDir)
{
    if (desc.type != LIGHT_TYPE_PROJECTOR || desc.projectorImagePath.empty())
        return;
    std::filesystem::path p(desc.projectorImagePath);
    if (!p.is_absolute() && !searchDir.empty())
        p = std::filesystem::path(searchDir) / p;
    desc.projectorImagePath = p.string();
    desc.projectorImage = scene.addProjectorImage(desc.projectorImagePath);
}

inline void resolveIes(Scene& scene, Scene::UniformLightDesc& desc, const std::string& searchDir)
{
    if (desc.iesPath.empty())
        return;
    std::filesystem::path p(desc.iesPath);
    if (!p.is_absolute())
        p = std::filesystem::path(searchDir) / p;
    Scene::IesProfile profile;
    if (loadIesProfile(p.string(), profile))
    {
        desc.iesProfile = scene.addIesProfile(std::move(profile));
        desc.iesPath = p.string();
    }
}

} // namespace oka::lightjson
