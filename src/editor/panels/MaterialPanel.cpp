#include "../EditorApp.h"

#include "imgui.h"

#include <algorithm>
#include <cmath>
#include <string>
#include <cstring>

namespace oka
{
namespace
{

// ImGui::InputText edits a fixed char buffer in place, so the path has to be
// copied into one. Truncates instead of overrunning: a path longer than the
// field is the field's problem.
template <size_t N>
void toEditBuffer(const std::string& src, char (&dst)[N])
{
    const size_t n = std::min(src.size(), N - 1);
    std::memcpy(dst, src.data(), n);
    dst[n] = '\0';
}

/// A parameter that a texture may be driving.
///
/// When it is, the widget is disabled rather than hidden. MaterialX semantics are
/// *replace*, not multiply -- a map on base_color leaves the constant beside it
/// doing nothing -- so a live slider under a map would be a lie the user only
/// discovers by dragging it and watching nothing happen. Disabled with the file
/// named in the tooltip says the same thing truthfully.
bool openpbrFloat(const char* label, float* value, float lo, float hi, const std::string& mapPath, const char* tip = nullptr)
{
    const bool mapped = !mapPath.empty();
    ImGui::BeginDisabled(mapped);
    const bool changed = ImGui::SliderFloat(label, value, lo, hi) && !mapped;
    ImGui::EndDisabled();
    // Asked before anything else is drawn: SameLine() moves the cursor but
    // IsItemHovered() follows the *last item*, so a tag drawn first would steal
    // the tooltip from the widget it labels.
    if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
    {
        ImGui::SetTooltip("%s", mapped ? ("Driven by " + mapPath).c_str() : (tip ? tip : label));
    }
    if (mapped)
    {
        ImGui::SameLine();
        ImGui::TextDisabled("[map]");
    }
    return changed;
}

bool openpbrColor(const char* label, OpenPBRColor* c, const std::string& mapPath, const char* tip = nullptr)
{
    const bool mapped = !mapPath.empty();
    ImGui::BeginDisabled(mapped);
    const bool changed = ImGui::ColorEdit3(label, &c->r) && !mapped;
    ImGui::EndDisabled();
    if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
    {
        ImGui::SetTooltip("%s", mapped ? ("Driven by " + mapPath).c_str() : (tip ? tip : label));
    }
    if (mapped)
    {
        ImGui::SameLine();
        ImGui::TextDisabled("[map]");
    }
    return changed;
}

/// A lobe: its weight, then everything the weight gates.
///
/// Returns whether the section is open *and* live. A layered model is hard to
/// read as forty sliders in a row -- what a reader wants first is which of the
/// eight lobes are switched on at all -- so the weight comes before the header's
/// contents and greys them out at zero.
/// "0.35", or "off". Two decimals by hand rather than a formatter: snprintf's
/// return value is one this build refuses to let go unchecked, and a header
/// label is not worth an <format> dependency.
std::string weightText(float weight)
{
    if (!(weight > 0.0f))
    {
        return "off";
    }
    const int hundredths = static_cast<int>(std::lround(weight * 100.0f));
    const int whole = hundredths / 100;
    const int frac = hundredths % 100;
    return std::to_string(whole) + "." + (frac < 10 ? "0" : "") + std::to_string(frac);
}

bool openpbrLobeHeader(const char* label, float weight)
{
    // The weight goes *into* the header's text, not beside it. A CollapsingHeader
    // spans the full width, so SameLine() after one puts the next item past the
    // right edge -- the state has to be readable with the section shut, which is
    // the whole reason it is here.
    //
    // The ### suffix pins the widget's identity to the label while the visible
    // text changes with the weight; without it every drag would look like a new
    // header to ImGui and the section would snap shut.
    const std::string title = std::string(label) + "  -  " + weightText(weight) + "###" + label;
    return ImGui::CollapsingHeader(title.c_str());
}

const char* const kSlotNames[MAX_OPENPBR_TEXTURES] = {
    "base_color",       "base_metalness",     "specular_roughness", "specular_color",  "specular_aniso",
    "coat_weight",      "coat_roughness",     "coat_color",         "fuzz_weight",     "fuzz_roughness",
    "emission_color",   "transmission_color", "subsurface_color",   "geometry_normal", "geometry_coat_normal",
    "geometry_opacity", "subsurface_weight",  "subsurface_radius",  "fuzz_color",
};

/// The OpenPBR parameter set, grouped the way the specification groups it.
bool drawOpenPBR(Scene::MaterialDescription& desc)
{
    OpenPBRParams& o = desc.openpbr;
    const auto& tex = desc.openpbrTexPaths;
    bool changed = false;

    if (openpbrLobeHeader("Base", o.base_weight))
    {
        changed |= openpbrFloat(
            "Weight##base", &o.base_weight, 0.0f, 1.0f, {}, "How much of the dielectric base is present at all.");
        changed |= openpbrColor("Color##base", &o.base_color, tex[OPENPBR_TEX_BASE_COLOR]);
        changed |= openpbrFloat("Metalness", &o.base_metalness, 0.0f, 1.0f, tex[OPENPBR_TEX_BASE_METALNESS],
                                "Blends the base from dielectric to conductor.");
        changed |= openpbrFloat("Diffuse roughness", &o.base_diffuse_roughness, 0.0f, 1.0f, {},
                                "Retroreflection in the diffuse lobe. glTF has no equivalent, so a converted "
                                "material always reads 0 here.");
    }

    if (openpbrLobeHeader("Specular", o.specular_weight))
    {
        changed |= openpbrFloat("Weight##spec", &o.specular_weight, 0.0f, 1.0f, {});
        changed |= openpbrColor("Color##spec", &o.specular_color, tex[OPENPBR_TEX_SPECULAR_COLOR]);
        changed |=
            openpbrFloat("Roughness##spec", &o.specular_roughness, 0.0f, 1.0f, tex[OPENPBR_TEX_SPECULAR_ROUGHNESS]);
        changed |= openpbrFloat("IOR##spec", &o.specular_ior, 1.0f, 3.0f, {});
        changed |= openpbrFloat(
            "Anisotropy##spec", &o.specular_roughness_anisotropy, 0.0f, 1.0f, tex[OPENPBR_TEX_SPECULAR_ANISOTROPY]);
    }

    if (openpbrLobeHeader("Coat", o.coat_weight))
    {
        changed |= openpbrFloat("Weight##coat", &o.coat_weight, 0.0f, 1.0f, tex[OPENPBR_TEX_COAT_WEIGHT]);
        changed |= openpbrColor("Color##coat", &o.coat_color, tex[OPENPBR_TEX_COAT_COLOR]);
        changed |= openpbrFloat("Roughness##coat", &o.coat_roughness, 0.0f, 1.0f, tex[OPENPBR_TEX_COAT_ROUGHNESS]);
        changed |= openpbrFloat("IOR##coat", &o.coat_ior, 1.0f, 3.0f, {});
        changed |= openpbrFloat("Darkening", &o.coat_darkening, 0.0f, 1.0f, {},
                                "How much the coat absorbs into what is under it. 1 is the spec default, not "
                                "'off' -- glTF's clearcoat has no such term, so a converted coat is legitimately "
                                "darker underneath.");
    }

    if (openpbrLobeHeader("Fuzz", o.fuzz_weight))
    {
        changed |= openpbrFloat("Weight##fuzz", &o.fuzz_weight, 0.0f, 1.0f, tex[OPENPBR_TEX_FUZZ_WEIGHT]);
        changed |= openpbrColor("Color##fuzz", &o.fuzz_color, tex[OPENPBR_TEX_FUZZ_COLOR]);
        changed |= openpbrFloat("Roughness##fuzz", &o.fuzz_roughness, 0.0f, 1.0f, tex[OPENPBR_TEX_FUZZ_ROUGHNESS],
                                "A microflake layer, not glTF's Charlie sheen. Expect fabric to differ.");
    }

    if (openpbrLobeHeader("Transmission", o.transmission_weight))
    {
        changed |= openpbrFloat("Weight##trans", &o.transmission_weight, 0.0f, 1.0f, {});
        changed |= openpbrColor("Color##trans", &o.transmission_color, tex[OPENPBR_TEX_TRANSMISSION_COLOR],
                                "The colour a beam is left with after transmission_depth.");
        changed |= ImGui::DragFloat("Depth", &o.transmission_depth, 0.01f, 0.0f, 100.0f);
        if (ImGui::IsItemHovered())
        {
            ImGui::SetTooltip("0 means no medium: the colour above is a surface tint and nothing absorbs.");
        }
        changed |= openpbrColor("Scatter", &o.transmission_scatter, {});
        changed |= openpbrFloat("Dispersion", &o.transmission_dispersion_scale, 0.0f, 1.0f, {});
        changed |= ImGui::DragFloat("Abbe number", &o.transmission_dispersion_abbe_number, 0.5f, 1.0f, 100.0f);
    }

    if (openpbrLobeHeader("Subsurface", o.subsurface_weight))
    {
        changed |= openpbrFloat("Weight##sss", &o.subsurface_weight, 0.0f, 1.0f, tex[OPENPBR_TEX_SUBSURFACE_WEIGHT]);
        changed |= openpbrColor("Color##sss", &o.subsurface_color, tex[OPENPBR_TEX_SUBSURFACE_COLOR]);
        // Not down to zero. The random walk's extinction is 1/radius (sssSigmaT),
        // saved from dividing by zero only by an internal clamp to 1e-5, so a
        // radius of 0 means an extinction of 1e5: every subsurface path then
        // spends the whole MEDIUM_MAX_STEPS budget on sub-micron flights and the
        // image does not change for it. Measured on the Open Chess Set, radius
        // 1.0 -> 0.0 costs 19.8 -> 50.7 ms/sample and looks the same.
        changed |= ImGui::DragFloat("Radius", &o.subsurface_radius, 0.001f, 1e-4f, 100.0f, "%.4f");
        if (ImGui::IsItemHovered())
        {
            ImGui::SetTooltip(
                "Mean free path in world units. Small values are dense media and cost a long "
                "random walk.");
        }
        changed |= openpbrColor("Radius scale", &o.subsurface_radius_scale, tex[OPENPBR_TEX_SUBSURFACE_RADIUS],
                                "Per-channel tint on the mean free path: red usually travels furthest.");
        changed |= ImGui::SliderFloat("Scatter anisotropy", &o.subsurface_scatter_anisotropy, -1.0f, 1.0f);
    }

    if (openpbrLobeHeader("Thin film", o.thin_film_weight))
    {
        changed |= openpbrFloat("Weight##film", &o.thin_film_weight, 0.0f, 1.0f, {});
        changed |= ImGui::DragFloat("Thickness (um)", &o.thin_film_thickness, 0.01f, 0.0f, 10.0f);
        changed |= openpbrFloat("IOR##film", &o.thin_film_ior, 1.0f, 3.0f, {});
    }

    if (openpbrLobeHeader("Emission", o.emission_luminance))
    {
        changed |= ImGui::DragFloat("Luminance", &o.emission_luminance, 1.0f, 0.0f, 100000.0f);
        if (ImGui::IsItemHovered())
        {
            ImGui::SetTooltip(
                "The specification calls this nits. Strelka passes it through 1:1 into the units "
                "its lights are measured in -- see MetalMaterials.mm.");
        }
        changed |= openpbrColor("Color##em", &o.emission_color, tex[OPENPBR_TEX_EMISSION_COLOR]);
    }

    if (ImGui::CollapsingHeader("Geometry"))
    {
        bool thin = o.geometry_thin_walled != 0u;
        if (ImGui::Checkbox("Thin walled", &thin))
        {
            o.geometry_thin_walled = thin ? 1u : 0u;
            changed = true;
        }
        if (ImGui::IsItemHovered())
        {
            ImGui::SetTooltip(
                "A thin-walled surface encloses nothing, so its interior medium is skipped "
                "entirely.");
        }
        changed |= openpbrFloat("Opacity", &o.geometry_opacity, 0.0f, 1.0f, tex[OPENPBR_TEX_GEOMETRY_OPACITY],
                                "Resolved by the renderer before shading; the BSDF does not consume it.");
    }

    // The maps, listed rather than edited: a path here is set by the .mtlx or the
    // sidecar the material came from, and typing a new one would silently
    // disagree with that file the next time the scene is loaded.
    int mapped = 0;
    for (uint32_t i = 0; i < MAX_OPENPBR_TEXTURES; ++i)
    {
        mapped += desc.openpbrTexPaths[i].empty() ? 0 : 1;
    }
    const std::string texTitle = "Textures  -  " + std::to_string(mapped) + "/" +
                                 std::to_string(static_cast<int>(MAX_OPENPBR_TEXTURES)) + "###Textures";
    if (ImGui::CollapsingHeader(texTitle.c_str()))
    {
        if (mapped == 0)
        {
            ImGui::TextDisabled("No maps. Every parameter above is a constant.");
        }
        for (uint32_t i = 0; i < MAX_OPENPBR_TEXTURES; ++i)
        {
            if (desc.openpbrTexPaths[i].empty())
            {
                continue;
            }
            ImGui::BulletText("%s", kSlotNames[i]);
            ImGui::SameLine();
            ImGui::TextDisabled("%s", desc.openpbrTexPaths[i].c_str());
        }
    }

    ImGui::Separator();
    if (ImGui::Button("Reset to OpenPBR defaults"))
    {
        desc.openpbr = openpbr_make_default_params();
        changed = true;
    }
    if (ImGui::IsItemHovered())
    {
        ImGui::SetTooltip(
            "The 1.1.1 specification defaults. A zeroed block is not one of them: it has no "
            "refractive index and a degenerate anisotropy frame.");
    }
    return changed;
}

} // namespace

void EditorApp::drawMaterialPanel()
{
    if (!ImGui::Begin("Materials", &m_showMaterials))
    {
        ImGui::End();
        return;
    }

    if (m_selectedInstanceId != kInvalidIndex && m_selectedInstanceId < m_scene->getInstances().size())
    {
        const auto& inst = m_scene->getInstances()[m_selectedInstanceId];
        if (inst.type == Instance::Type::eMesh)
            m_selectedMaterialId = inst.mMaterialId;
    }

    auto& materials = m_scene->getMaterials();
    if (materials.empty())
    {
        ImGui::TextUnformatted("No materials in scene.");
        ImGui::End();
        return;
    }

    if (m_selectedMaterialId >= materials.size())
        m_selectedMaterialId = 0;

    if (ImGui::BeginCombo("Material", materials[m_selectedMaterialId].name.c_str()))
    {
        for (uint32_t i = 0; i < materials.size(); ++i)
        {
            const bool selected = (i == m_selectedMaterialId);
            const char* name = materials[i].name.empty() ? "(unnamed)" : materials[i].name.c_str();
            if (ImGui::Selectable(name, selected))
                m_selectedMaterialId = i;
            if (selected)
                ImGui::SetItemDefaultFocus();
        }
        ImGui::EndCombo();
    }

    Scene::MaterialDescription desc = materials[m_selectedMaterialId];
    bool changed = false;

    // Which model this material shades with, said out loud. The two have
    // different parameter sets and the panel below is a different panel; without
    // this the user is left to infer it from which sliders appeared.
    const bool isOpenPBR = desc.params.material_type == MATERIAL_TYPE_OPENPBR;
    ImGui::TextDisabled("Model");
    ImGui::SameLine();
    if (isOpenPBR)
    {
        ImGui::TextColored(ImVec4(0.45f, 0.78f, 1.0f, 1.0f), "OpenPBR Surface 1.1.1");
    }
    else
    {
        ImGui::TextUnformatted("glTF metallic-roughness");
    }
    if (ImGui::IsItemHovered())
    {
        ImGui::SetTooltip(
            "%s", isOpenPBR ?
                      "Authored by a <stem>_openpbr.json sidecar, a <stem>.mtlx document, or converted scene-wide by "
                      "render/material/model." :
                      "The model every glTF scene loads into.");
    }
    ImGui::Separator();

    if (isOpenPBR)
    {
        changed |= drawOpenPBR(desc);
    }
    else
    {
        changed |= ImGui::ColorEdit3("Base Color", &desc.params.base_color.x);
        changed |= ImGui::DragFloat("Metallic", &desc.params.metallic, 0.01f, 0.0f, 1.0f);
        changed |= ImGui::DragFloat("Roughness", &desc.params.roughness, 0.01f, 0.0f, 1.0f);
        changed |= ImGui::ColorEdit3("Emission", &desc.params.emission.x);
        changed |= ImGui::DragFloat("Emission Strength", &desc.params.emission_strength, 0.1f, 0.0f, 1000.0f);
        changed |= ImGui::DragFloat("IOR", &desc.params.ior, 0.01f, 1.0f, 3.0f);

        char pathBuf[512];
        toEditBuffer(desc.baseColorTexPath, pathBuf);
        if (ImGui::InputText("Base Color Tex", pathBuf, sizeof(pathBuf)))
        {
            desc.baseColorTexPath = pathBuf;
            changed = true;
        }
    }

    {
        ImGui::SeparatorText("Environment");
        Scene::EnvLightDesc env = m_scene->getEnvLight().value_or(Scene::EnvLightDesc{});
        bool envChanged = false;
        char envPath[512];
        toEditBuffer(env.texturePath, envPath);
        if (ImGui::InputText("HDR path", envPath, sizeof(envPath)))
        {
            env.texturePath = envPath;
            envChanged = true;
        }
        envChanged |= ImGui::DragFloat("Env Intensity", &env.intensity, 0.05f, 0.0f, 100.0f);
        envChanged |= ImGui::ColorEdit3("Env Color", &env.color.x);
        envChanged |= ImGui::DragFloat("Env Rotation Y", &env.rotationY, 0.5f);
        if (envChanged)
        {
            m_scene->setEnvLight(env);
            markDocumentDirty();
        }
    }

    if (changed)
    {
        pushUndoMaterial(m_selectedMaterialId);
        m_scene->setMaterial(m_selectedMaterialId, desc);
        markDocumentDirty();
    }

    ImGui::End();
}

} // namespace oka
