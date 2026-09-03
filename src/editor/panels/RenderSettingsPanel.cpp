#include "../EditorApp.h"
#include "../editor_denoiser_ui.h"
#include "../editor_frame_budget.h"

#include <strelka/display/output_policy.h>

#include "imgui.h"
#include "ImGuiFileDialog.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <string>
#include <utility>
#include <vector>

namespace oka
{
namespace
{
/// Label/value fact table for both backends' Capabilities section. Replaces
/// TextDisabled lines joined with " | ", which ran off the panel edge.
void drawFactTable(const char* id, const std::vector<std::pair<const char*, std::string>>& rows)
{
    if (!ImGui::BeginTable(id, 2, ImGuiTableFlags_None))
    {
        return;
    }
    ImGui::TableSetupColumn("##label", ImGuiTableColumnFlags_WidthFixed, 108.0f);
    ImGui::TableSetupColumn("##value", ImGuiTableColumnFlags_WidthStretch);
    for (const auto& [label, value] : rows)
    {
        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::TextDisabled("%s", label);
        ImGui::TableSetColumnIndex(1);
        ImGui::TextWrapped("%s", value.c_str());
    }
    ImGui::EndTable();
}

/// Vulkan/OptiX output settings: an HDR10 swapchain the application negotiates,
/// with absolute nits for the metadata it attaches to it.
void drawSwapchainOutputSettings(SettingsManager& settings,
                                 const display_output::DisplayCapabilities& capabilities)
{
    const char* const outputModeItems[] = { "Auto", "HDR10", "SDR" };
    uint32_t storedMode = 0;
    int outputMode = 0;
    int n = 0;
    bool isSelected = false;
    bool vrrEnabled = false;
    float paperWhite = NAN;
    float peakNits = NAN;
    const char *vrrStatus = nullptr;

    storedMode = settings.getAs<uint32_t>("render/post/outputMode");
    outputMode = static_cast<int>(std::min(storedMode, static_cast<uint32_t>(display_output::OutputMode::SDR)));
    if (ImGui::BeginCombo("Dynamic range", outputModeItems[outputMode]))
    {
        for (n = 0; n < IM_ARRAYSIZE(outputModeItems); ++n)
        {
            isSelected = outputMode == n;
            if (ImGui::Selectable(outputModeItems[n], isSelected))
            {
                outputMode = n;
                settings.setAs<uint32_t>("render/post/outputMode", static_cast<uint32_t>(n));
            }
            if (isSelected)
            {
                ImGui::SetItemDefaultFocus();
            }
        }
        ImGui::EndCombo();
    }

    paperWhite = settings.getAs<float>("render/post/paperWhiteNits");
    peakNits = settings.getAs<float>("render/post/peakNits");
    ImGui::BeginDisabled(outputMode == static_cast<int>(display_output::OutputMode::SDR));
    if (ImGui::DragFloat("Paper white", &paperWhite, 1.0f, 80.0f, 500.0f, "%.0f nits"))
    {
        peakNits = std::max(peakNits, paperWhite);
        settings.setAs<float>("render/post/paperWhiteNits", paperWhite);
        settings.setAs<float>("render/post/peakNits", peakNits);
    }
    if (ImGui::DragFloat("HDR peak", &peakNits, 10.0f, paperWhite, 10000.0f, "%.0f nits",
                         ImGuiSliderFlags_Logarithmic))
    {
        settings.setAs<float>("render/post/peakNits", peakNits);
    }
    ImGui::EndDisabled();

    vrrEnabled = settings.getAs<bool>("display/vrr/enabled");
    if (ImGui::Checkbox("Variable refresh rate", &vrrEnabled))
    {
        settings.setAs<bool>("display/vrr/enabled", vrrEnabled);
    }

    vrrStatus = display_output::vrrStatusName(capabilities.vrrStatus);
    ImGui::SeparatorText("Capabilities");
    std::vector<std::pair<const char*, std::string>> facts = {
        { "HDR10", fmt::format("{}  (selected {}, metadata {})",
                               capabilities.output.hdr10 ? "supported" : "unavailable",
                               capabilities.output.hdrSelected ? "yes" : "no",
                               capabilities.output.hdrMetadata ? "supported" : "unavailable") },
        { "Present modes", fmt::format("FIFO{}{}{}", capabilities.present.fifoRelaxed ? ", FIFO_RELAXED" : "",
                                       capabilities.present.mailbox ? ", MAILBOX" : "",
                                       capabilities.present.immediate ? ", IMMEDIATE" : "") },
        { "Present wait", fmt::format("{}  (present ID {}, timing {})",
                                      capabilities.presentWait ? "available" : "unavailable",
                                      capabilities.presentId ? "available" : "unavailable",
                                      capabilities.displayTiming ? "available" : "unavailable") },
        { "VRR", fmt::format("{}  (current {:.3f} Hz, Vulkan FIFO baseline)", vrrStatus,
                             capabilities.currentRefreshRateHz) },
    };
    if (capabilities.minRefreshRateHz > 0.0f && capabilities.maxRefreshRateHz > 0.0f)
    {
        facts.emplace_back("Compositor VRR", fmt::format("{:.3f}-{:.3f} Hz", capabilities.minRefreshRateHz,
                                                          capabilities.maxRefreshRateHz));
    }
    drawFactTable("##swapchainCapabilities", facts);
}

/// Metal output settings.
///
/// Deliberately not the swapchain panel above with the words changed. macOS
/// gives an application no HDR10 surface to select and no absolute luminance to
/// target: the window server grants a *headroom*, a multiplier over SDR white
/// that moves with the brightness slider, the thermal state and what other
/// windows are asking for. So there is nothing here to set in nits, and the
/// choices that do exist -- how much of the granted headroom to use, and how the
/// layer presents -- have no counterpart on the Vulkan side.
void drawMetalOutputSettings(SettingsManager& settings,
                             const display_output::DisplayCapabilities& capabilities)
{
    struct ModeItem
    {
        display_output::OutputMode mode;
        const char* name;
        const char* help;
    };
    // Order matches display_output::OutputMode so the combo index is the stored
    // value; ReferenceHDR is last there for the Vulkan clamp, and last here
    // because it is the specialist entry.
    static const ModeItem kModes[] = {
        { display_output::OutputMode::Auto, "Auto",
          "Follow the display. Uses whatever headroom the window server currently\n"
          "grants, and falls back to plain SDR on a panel that grants none." },
        { display_output::OutputMode::HDR, "Extended (EDR)",
          "Always ask for extended-range output, even while the granted headroom\n"
          "is 1.0. Same result as Auto on a display that has headroom; the\n"
          "difference is that the layer keeps asking, so the image follows the\n"
          "headroom back up when it is restored." },
        { display_output::OutputMode::SDR, "SDR",
          "Clamp at SDR white and drop the layer to a plain sRGB colour space.\n"
          "This is the mode to compare against a screenshot, an EXR viewer or a\n"
          "second machine: everything above white is gone in all of them." },
        { display_output::OutputMode::ReferenceHDR, "Reference (XDR)",
          "Map the tone curve to the panel's reference headroom instead of the\n"
          "headroom handed to an ordinary window. Only on an XDR display in a\n"
          "reference preset -- pick one in System Settings > Displays before this\n"
          "entry becomes selectable." },
    };
    const uint32_t storedMode =
        std::min(settings.getAs<uint32_t>("render/post/outputMode"),
                 static_cast<uint32_t>(display_output::OutputMode::ReferenceHDR));
    int selectedItem = 0;
    int n = 0;
    bool supported = false;
    bool isSelected = false;
    float headroomLimit = NAN;
    float frameRateLimit = NAN;
    bool vsync = false;
    bool tripleBuffering = false;
    char referenceText[32] = "unavailable";

    for (n = 0; n < IM_ARRAYSIZE(kModes); ++n)
    {
        if (static_cast<uint32_t>(kModes[n].mode) == storedMode)
        {
            selectedItem = n;
        }
    }

    ImGui::TextDisabled("Display: %s",
                        capabilities.displayName.empty() ? "unknown" : capabilities.displayName.c_str());

    if (ImGui::BeginCombo("Dynamic range", kModes[selectedItem].name))
    {
        for (n = 0; n < IM_ARRAYSIZE(kModes); ++n)
        {
            supported = display_output::outputModeSupported(kModes[n].mode, capabilities.edr);
            isSelected = n == selectedItem;
            if (ImGui::Selectable(kModes[n].name, isSelected,
                                  supported ? ImGuiSelectableFlags_None : ImGuiSelectableFlags_Disabled))
            {
                settings.setAs<uint32_t>("render/post/outputMode", static_cast<uint32_t>(kModes[n].mode));
            }
            if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
            {
                ImGui::SetTooltip("%s%s", kModes[n].help,
                                  supported ? "" : "\n\nThis display does not offer it.");
            }
            if (isSelected)
            {
                ImGui::SetItemDefaultFocus();
            }
        }
        ImGui::EndCombo();
    }

    // A ceiling, not a target: the display still decides what it grants, this
    // only stops the tone curve from spending all of it. Worth having because a
    // panel that grants 16x makes an ordinary interior render look like a
    // lightbox, and because A/B against an SDR reference needs a fixed number
    // rather than one the compositor keeps moving.
    headroomLimit = settings.getAs<float>("display/edr/headroomLimit");
    ImGui::BeginDisabled(storedMode == static_cast<uint32_t>(display_output::OutputMode::SDR));
    if (ImGui::DragFloat("Headroom limit", &headroomLimit, 0.05f, 0.0f,
                         std::max(capabilities.edr.potentialHeadroom, 2.0f),
                         headroomLimit >= 1.0f ? "%.2fx" : "%.0f = display maximum"))
    {
        settings.setAs<float>("display/edr/headroomLimit", headroomLimit);
    }
    ImGui::EndDisabled();

    vsync = settings.getAs<bool>("display/vsync/enabled");
    if (ImGui::Checkbox("V-Sync", &vsync))
    {
        settings.setAs<bool>("display/vsync/enabled", vsync);
    }
    if (ImGui::IsItemHovered())
    {
        ImGui::SetTooltip(
            "CAMetalLayer displaySyncEnabled. Off presents as fast as the layer\n"
            "hands out drawables, which tears but takes the compositor out of a\n"
            "frame time measurement.");
    }

    tripleBuffering = settings.getAs<bool>("display/present/tripleBuffering");
    if (ImGui::Checkbox("Triple buffering", &tripleBuffering))
    {
        settings.setAs<bool>("display/present/tripleBuffering", tripleBuffering);
    }
    if (ImGui::IsItemHovered())
    {
        ImGui::SetTooltip(
            "Three drawables absorb a late frame; two save a frame of latency and\n"
            "make the editor feel more direct on a fast scene.");
    }

    frameRateLimit = settings.getAs<float>("display/present/fpsLimit");
    if (ImGui::DragFloat("Frame rate limit", &frameRateLimit, 1.0f, 0.0f,
                         std::max(capabilities.maxRefreshRateHz, 240.0f),
                         frameRateLimit >= 1.0f ? "%.0f fps" : "%.0f = display refresh"))
    {
        settings.setAs<float>("display/present/fpsLimit", std::max(frameRateLimit, 0.0f));
    }
    if (ImGui::IsItemHovered())
    {
        ImGui::SetTooltip(
            "Presented as a minimum frame duration, not a sleep. On a\n"
            "variable-refresh panel the window server answers it by dropping the\n"
            "panel to a matching rate, which is where the power saving comes from;\n"
            "on a fixed-rate one it just paces the presents.");
    }

    ImGui::SeparatorText("Capabilities");
    if (capabilities.edr.referenceHeadroom > 1.0f)
    {
        // Truncation would only shorten a cosmetic label, and snprintf still
        // terminates the buffer, so there is nothing here to recover from.
        (void)snprintf(referenceText, sizeof(referenceText), "%.2fx", capabilities.edr.referenceHeadroom);
    }
    const std::vector<std::pair<const char*, std::string>> facts = {
        { "EDR headroom", fmt::format("{:.2f}x now, {:.2f}x max", capabilities.edr.currentHeadroom,
                                      capabilities.edr.potentialHeadroom) },
        { "EDR reference", referenceText },
        { "Tone curve", fmt::format("{:.2f}x SDR white", capabilities.appliedHeadroom) },
        { "Colour space", fmt::format("{}{}", capabilities.colorSpaceName.empty() ? "unknown" :
                                                                                     capabilities.colorSpaceName.c_str(),
                                      capabilities.edr.wideGamut ? " (P3 capable)" : "") },
        { "Layer", capabilities.edr.edrRequested ? "extended sRGB, RGBA16F" : "sRGB, RGBA16F" },
        capabilities.vrrStatus == display_output::VrrStatus::Supported ?
            std::pair<const char*, std::string>{ "Refresh", fmt::format("variable {:.1f}-{:.1f} Hz (up to {:.0f} fps)",
                                                                        capabilities.minRefreshRateHz,
                                                                        capabilities.maxRefreshRateHz,
                                                                        capabilities.currentRefreshRateHz) } :
            std::pair<const char*, std::string>{ "Refresh", fmt::format("fixed {:.1f} Hz", capabilities.maxRefreshRateHz) },
        { "Present", fmt::format("{} drawables  (vsync {}, {})", capabilities.maxDrawableCount,
                                 capabilities.displaySync ? "on" : "off",
                                 capabilities.frameRateLimitHz > 0.0f ? "rate limited" : "display rate") },
    };
    drawFactTable("##metalCapabilities", facts);
}

void drawDisplayOutputSettings(SettingsManager& settings, const Display& display)
{
    display_output::DisplayCapabilities capabilities;

    if (!ImGui::TreeNodeEx("Display output", ImGuiTreeNodeFlags_DefaultOpen))
    {
        return;
    }

    // Which controls exist is a property of the backend, not a preference. The
    // swapchain panel used to be drawn on macOS too, where every row it reports
    // reads "unavailable" -- not because the display cannot do it, but because
    // none of it is a Metal concept.
    capabilities = display.getOutputCapabilities();
    if (capabilities.backend == display_output::DisplayBackend::Metal)
    {
        drawMetalOutputSettings(settings, capabilities);
    }
    else
    {
        drawSwapchainOutputSettings(settings, capabilities);
    }
    ImGui::TreePop();
}
} // namespace

void EditorApp::drawRenderSettingsPanel()
{
    if (!ImGui::Begin("Render Settings:"))
    {
        ImGui::End();
        return;
    }

    if (ImGui::BeginTabBar("RenderSettingsTabs"))
    {

        if (ImGui::BeginTabItem("Display"))
        {
            drawDisplayOutputSettings(*m_settingsManager, *m_display);

            if (ImGui::TreeNode("Display tonemap"))
            {
                const char* const tonemapItems[] = { "None", "Reinhard", "ACES", "Filmic" };
                int currentTonemapItemId =
                    (int)std::min(m_settingsManager->getAs<uint32_t>("render/pt/tonemapperType"), 3u);
                if (ImGui::BeginCombo("Operator", tonemapItems[currentTonemapItemId]))
                {
                    for (int n = 0; n < IM_ARRAYSIZE(tonemapItems); n++)
                    {
                        const bool is_selected = (currentTonemapItemId == n);
                        if (ImGui::Selectable(tonemapItems[n], is_selected))
                        {
                            currentTonemapItemId = n;
                            m_settingsManager->setAs<uint32_t>("render/pt/tonemapperType", (uint32_t)n);
                        }
                        if (is_selected)
                        {
                            ImGui::SetItemDefaultFocus();
                        }
                    }
                    ImGui::EndCombo();
                }

                auto gamma = m_settingsManager->getAs<float>("render/post/gamma");
                if (ImGui::DragFloat("Gamma", &gamma, 0.05f, 0.0f, 5.0f, "%.2f"))
                {
                    m_settingsManager->setAs<float>("render/post/gamma", gamma);
                }
                if (ImGui::IsItemHovered())
                {
                    ImGui::SetTooltip("0 = off; default 2.4 is an sRGB-like transfer, not a pure power.");
                }

                const float maxEdr = m_settingsManager->getAs<float>("render/post/tonemapper/maxEDR");
                ImGui::TextDisabled("Display max EDR %.2f (tone-map shoulder follows screen headroom)", maxEdr);

                ImGui::TreePop();
            }

            ImGui::EndTabItem();
        }

        if (ImGui::BeginTabItem("Quality"))
        {
            ImGui::SeparatorText("Preview");
            {
                uint32_t previewWidth = m_settingsManager->getAs<uint32_t>("render/width");
                uint32_t previewHeight = m_settingsManager->getAs<uint32_t>("render/height");
                const int preset = editor_viewport::findPreset(previewWidth, previewHeight);
                const int presetCount = static_cast<int>(editor_viewport::kPreviewPresets.size());
                static bool customSelected = false;
                static uint32_t previousWidth = previewWidth;
                static uint32_t previousHeight = previewHeight;
                if ((previewWidth != previousWidth || previewHeight != previousHeight) && preset >= 0)
                {
                    customSelected = false;
                }
                int selectedPreset = !customSelected && preset >= 0 ? preset : presetCount;
                const char* const presetNames[] = {
                    editor_viewport::kPreviewPresets[0].label,
                    editor_viewport::kPreviewPresets[1].label,
                    editor_viewport::kPreviewPresets[2].label,
                    editor_viewport::kPreviewPresets[3].label,
                    "Custom",
                };
                if (ImGui::Combo("Preset", &selectedPreset, presetNames, IM_ARRAYSIZE(presetNames)))
                {
                    customSelected = selectedPreset == presetCount;
                    if (!customSelected && selectedPreset >= 0 && selectedPreset < presetCount)
                    {
                        const editor_viewport::PreviewPreset& selected = editor_viewport::kPreviewPresets[selectedPreset];
                        requestPreviewResolution(selected.width, selected.height);
                        previewWidth = selected.width;
                        previewHeight = selected.height;
                    }
                }

                if (selectedPreset == presetCount)
                {
                    static bool lockAspect = true;
                    int customWidth = static_cast<int>(previewWidth);
                    int customHeight = static_cast<int>(previewHeight);
                    const float aspect =
                        previewHeight > 0 ? static_cast<float>(previewWidth) / static_cast<float>(previewHeight) : 1.0f;
                    if (ImGui::InputInt("Width", &customWidth))
                    {
                        const uint32_t width = editor_viewport::clampPreviewDimension(customWidth);
                        const uint32_t height = lockAspect ? editor_viewport::clampPreviewDimension(static_cast<int>(
                                                                 std::lround(static_cast<float>(width) / aspect))) :
                                                             previewHeight;
                        requestPreviewResolution(width, height);
                    }
                    if (ImGui::InputInt("Height", &customHeight))
                    {
                        const uint32_t height = editor_viewport::clampPreviewDimension(customHeight);
                        const uint32_t width = lockAspect ? editor_viewport::clampPreviewDimension(static_cast<int>(
                                                                std::lround(static_cast<float>(height) * aspect))) :
                                                            previewWidth;
                        requestPreviewResolution(width, height);
                    }
                    ImGui::Checkbox("Lock aspect ratio", &lockAspect);
                    ImGui::SameLine();
                    if (ImGui::Button("Swap"))
                    {
                        requestPreviewResolution(previewHeight, previewWidth);
                    }
                }

                ImGui::TextDisabled("Viewport resize changes presentation only");
                previousWidth = m_settingsManager->getAs<uint32_t>("render/width");
                previousHeight = m_settingsManager->getAs<uint32_t>("render/height");
            }

            ImGui::SeparatorText("Sampling");
            {
                const char* const rectlightSamplingMethodItems[] = { "Uniform", "Advanced" };
                int currentRectlightSamplingMethodItemId = static_cast<int>(
                    std::min(m_settingsManager->getAs<uint32_t>("render/pt/rectLightSamplingMethod"), 1u));
                if (ImGui::BeginCombo(
                        "Rect Light Sampling", rectlightSamplingMethodItems[currentRectlightSamplingMethodItemId]))
                {
                    for (const auto& item : rectlightSamplingMethodItems)
                    {
                        const bool is_selected =
                            (item == rectlightSamplingMethodItems[currentRectlightSamplingMethodItemId]);
                        if (ImGui::Selectable(item, is_selected))
                        {
                            currentRectlightSamplingMethodItemId = static_cast<int>(&item - rectlightSamplingMethodItems);
                            m_settingsManager->setAs<uint32_t>(
                                "render/pt/rectLightSamplingMethod",
                                static_cast<uint32_t>(currentRectlightSamplingMethodItemId));
                        }
                        if (is_selected)
                        {
                            ImGui::SetItemDefaultFocus();
                        }
                    }
                    ImGui::EndCombo();
                }

                const char* const samplerTypeItems[] = { "Halton", "PCG", "Sobol (Owen)", "Sobol + blue noise",
                                                         "Hybrid (blue noise -> Sobol)" };
                // Read back rather than remembered in a static: the default is set in
                // loadSettings, and a static starting at zero showed "Halton" no matter
                // what was actually running.
                int currentSamplerTypeId = (int)std::min(m_settingsManager->getAs<uint32_t>("render/pt/samplerType"), 4u);
                if (ImGui::BeginCombo("Sampler", samplerTypeItems[currentSamplerTypeId]))
                {
                    for (const auto& item : samplerTypeItems)
                    {
                        const bool is_selected = (item == samplerTypeItems[currentSamplerTypeId]);
                        if (ImGui::Selectable(item, is_selected))
                        {
                            currentSamplerTypeId = (int)(&item - samplerTypeItems);
                            m_settingsManager->setAs<uint32_t>("render/pt/samplerType", currentSamplerTypeId);
                        }
                        if (is_selected)
                        {
                            ImGui::SetItemDefaultFocus();
                        }
                    }
                    ImGui::EndCombo();
                }
                if (currentSamplerTypeId == 0)
                {
                    ImGui::TextDisabled("Halton aliases its bases every 32 dimensions; error stalls past ~512 spp.");
                }
                if (currentSamplerTypeId == 4)
                {
                    auto bnSwitch = m_settingsManager->getAs<uint32_t>("render/pt/blueNoiseSwitchSpp");
                    if (ImGui::SliderInt("Blue-noise samples", (int*)&bnSwitch, 0, 256))
                    {
                        m_settingsManager->setAs<uint32_t>("render/pt/blueNoiseSwitchSpp", bnSwitch);
                    }
                    if (ImGui::IsItemHovered())
                    {
                        ImGui::SetTooltip(
                            "Samples drawn from the blue-noise sequence before handing over to\n"
                            "per-pixel scrambling. Blue noise looks cleaner at low sample counts;\n"
                            "scrambling converges faster past a few dozen.");
                    }
                }

                ImGui::SeparatorText("Denoiser");

                // One choice, not two checkboxes -- and this backend's choices, not the
                // other backend's.
                //
                // Denoising and upscaling are alternatives on both: MetalFX sends a frame
                // through the spatial scaler or the temporal denoised one, and the OptiX
                // plan has upscaling imply denoising because nothing in it scales without
                // also running the network. As separate toggles they offered four states,
                // two of which meant the same thing and none of which said so.
                //
                // Which states exist, what they are called, whether the render scale is a
                // slider or follows from the mode, and what the fallback warning means
                // are all the backend's answer to give -- see editor_denoiser_ui.h. This
                // panel used to hard-code MetalFX's answers and show them over OptiX,
                // where the name was wrong and the scale slider did nothing.
                const editor_denoiser::Ui fx = editor_denoiser::uiFor(m_render->denoiserKind());
                const float requestedScale = m_settingsManager->getAs<float>("render/pt/upscaleFactor");
                const bool denoiseSetting = m_settingsManager->getAs<bool>("render/pt/denoise");
                const bool upscaleSetting = m_settingsManager->getAs<bool>("render/pt/enableUpscale");
                // Keep the remembered index and what is actually running in step, every
                // frame rather than once. A mode index outliving the list it indexed is
                // how the combo came to show one thing while the renderer ran another: it
                // displayed whatever sat at that index, and the next click picked
                // something nobody asked for. The settings move without the panel too --
                // the frame-budget button, the benchmark drivers, STRELKA_DENOISE.
                if (fx.modeCount > 0 && (!mDenoiseModeInitialized || mDenoiseModeIndex >= fx.modeCount ||
                                         !editor_denoiser::settingsMatchMode(
                                             fx, mDenoiseModeIndex, denoiseSetting, upscaleSetting, requestedScale)))
                {
                    mDenoiseModeIndex = editor_denoiser::modeIndexFromSettings(fx, denoiseSetting, upscaleSetting);
                    mDenoiseModeInitialized = true;
                }
                const editor_denoiser::Mode fxMode = editor_denoiser::modeAt(fx, mDenoiseModeIndex);
                const bool denoiserOn = fxMode.denoise || fxMode.upscale;

                if (!editor_denoiser::hasDenoiser(fx))
                {
                    ImGui::TextDisabled("This backend has no denoiser");
                }
                else
                {
                    if (ImGui::BeginCombo(fx.title, editor_denoiser::modeAt(fx, mDenoiseModeIndex).label))
                    {
                        for (int n = 0; n < fx.modeCount; n++)
                        {
                            const bool is_selected = (mDenoiseModeIndex == n);
                            if (ImGui::Selectable(fx.modes[n].label, is_selected) && mDenoiseModeIndex != n)
                            {
                                mDenoiseModeIndex = n;
                                m_settingsManager->setAs<bool>("render/pt/denoise", fx.modes[n].denoise);
                                // The denoiser is a scaler too: it needs the reduced-
                                // resolution render whenever the mode asks for one, and
                                // nothing else does.
                                m_settingsManager->setAs<bool>(
                                    "render/pt/enableUpscale", editor_denoiser::shouldUpscale(fx, n, requestedScale));
                                m_render->resetTemporalHistory();
                            }
                            if (is_selected)
                            {
                                ImGui::SetItemDefaultFocus();
                            }
                        }
                        ImGui::EndCombo();
                    }

                    if (denoiserOn && fx.modeHint != nullptr)
                    {
                        ImGui::TextDisabled("%s", fx.modeHint);
                    }

                    // Temporal is a property of the network on OptiX rather than a mode of
                    // its own: both the denoise-only and the 2x model have a temporal
                    // variant, so it is one switch instead of a doubled list.
                    if (fx.temporalToggle && denoiserOn)
                    {
                        bool temporal = m_settingsManager->getAs<uint32_t>("render/pt/upscaleMode") == 1u;
                        if (ImGui::Checkbox("Temporal", &temporal))
                        {
                            m_settingsManager->setAs<uint32_t>("render/pt/upscaleMode", temporal ? 1u : 0u);
                            m_render->resetTemporalHistory();
                        }
                        ImGui::SameLine();
                        ImGui::BeginDisabled();
                        ImGui::TextUnformatted(temporal ? "(reprojects the previous frame)" :
                                                          "(each frame denoised alone)");
                        ImGui::EndDisabled();
                    }

                    if (fx.playbackMotionBlurToggle && fxMode.denoise)
                    {
                        bool playbackBlur = m_settingsManager->getAs<bool>("render/pt/denoisePlaybackMotionBlur");
                        if (ImGui::Checkbox("Path-traced playback blur", &playbackBlur))
                        {
                            m_settingsManager->setAs<bool>("render/pt/denoisePlaybackMotionBlur", playbackBlur);
                            m_render->resetTemporalHistory();
                        }
                        ImGui::SameLine();
                        ImGui::BeginDisabled();
                        ImGui::TextUnformatted(playbackBlur ? "(uses SPP per frame)" : "(stable shutter-close guides)");
                        ImGui::EndDisabled();
                    }

                    if (denoiserOn && fx.freeRenderScale)
                    {
                        float factor = requestedScale;
                        if (ImGui::SliderFloat("PT scale inside preview", &factor, 0.25f, 1.0f, "%.2f"))
                        {
                            m_settingsManager->setAs<float>("render/pt/upscaleFactor", factor);
                            m_settingsManager->setAs<bool>(
                                "render/pt/enableUpscale", editor_denoiser::shouldUpscale(fx, mDenoiseModeIndex, factor));
                            m_render->resetTemporalHistory();
                        }
                        ImGui::SameLine();
                        ImGui::BeginDisabled();
                        const char* const scaleStatus =
                            factor < 1.0f ?
                                "(rendering below display resolution)" :
                                (fxMode.denoise ? "(denoising at 1:1)" : "(inactive at 1:1; lower scale to enable)");
                        ImGui::TextUnformatted(scaleStatus);
                        ImGui::EndDisabled();
                    }
                }

                const uint32_t displayWidth = m_settingsManager->getAs<uint32_t>("render/width");
                const uint32_t displayHeight = m_settingsManager->getAs<uint32_t>("render/height");
                const editor_denoiser::Resolution previewRes =
                    editor_denoiser::resolution(fx, mDenoiseModeIndex, requestedScale, displayWidth, displayHeight);
                ImGui::TextDisabled("PT internal: %u x %u", previewRes.pathTraceWidth, previewRes.pathTraceHeight);
                ImGui::TextDisabled("Preview output: %u x %u", previewRes.outputWidth, previewRes.outputHeight);
                if (m_render->denoiserFallbackActive())
                {
                    ImGui::TextColored(ImVec4(1.0f, 0.75f, 0.25f, 1.0f), "%s", fx.fallbackMessage);
                }
                const double lastGpuMs = m_render->getLastRenderTimeMs();
                if (lastGpuMs > editor_frame_budget::kInteractiveBudgetMs)
                {
                    ImGui::TextColored(ImVec4(1.0f, 0.75f, 0.25f, 1.0f),
                                       "Last PT frame: %.0f ms (interactive budget: %.0f ms)", lastGpuMs,
                                       editor_frame_budget::kInteractiveBudgetMs);
                    // The scale the frame was *actually* traced at, not the one the
                    // slider holds: the budget divides a measured GPU time by a pixel
                    // count, and on a fixed-ratio backend the slider is not that count.
                    const float tracedScale = editor_denoiser::appliedScale(fx, mDenoiseModeIndex, requestedScale);
                    const editor_frame_budget::RenderSettingsSnapshot current{
                        displayWidth,
                        displayHeight,
                        tracedScale < 1.0f,
                        tracedScale,
                    };
                    const editor_frame_budget::FrameSample sample = editor_frame_budget::sampleFrom(lastGpuMs, current);
                    if (fx.freeRenderScale)
                    {
                        const float suggestedScale =
                            editor_frame_budget::recommendedScale(sample, displayWidth, displayHeight);
                        // Scaling, not denoising: the cheapest way to buy frame time, and
                        // the one that does not depend on a history the camera is about to
                        // invalidate anyway.
                        const int scalingMode = editor_denoiser::modeIndexFromSettings(fx, false, true);
                        const std::string label = fmt::format("Lower PT scale to {:.2f}", suggestedScale);
                        if (ImGui::Button(label.c_str()))
                        {
                            m_settingsManager->setAs<bool>("render/pt/denoise", fx.modes[scalingMode].denoise);
                            m_settingsManager->setAs<uint32_t>("render/pt/upscaleMode", 0);
                            m_settingsManager->setAs<float>("render/pt/upscaleFactor", suggestedScale);
                            m_settingsManager->setAs<bool>(
                                "render/pt/enableUpscale",
                                editor_denoiser::shouldUpscale(fx, scalingMode, suggestedScale));
                            mDenoiseModeIndex = scalingMode;
                            mDenoiseModeInitialized = true;
                            m_render->resetTemporalHistory();
                        }
                    }
                    else if (editor_denoiser::hasDenoiser(fx))
                    {
                        // Nothing to lower: this backend's only lever is its fixed ratio,
                        // so the offer is to switch it on rather than to pick a number.
                        const int scalingMode = editor_denoiser::modeIndexFromSettings(fx, true, true);
                        const std::string label = fmt::format("Switch to \"{}\"", fx.modes[scalingMode].label);
                        ImGui::BeginDisabled(mDenoiseModeIndex == scalingMode);
                        if (ImGui::Button(label.c_str()))
                        {
                            m_settingsManager->setAs<bool>("render/pt/denoise", fx.modes[scalingMode].denoise);
                            m_settingsManager->setAs<bool>(
                                "render/pt/enableUpscale",
                                editor_denoiser::shouldUpscale(fx, scalingMode, requestedScale));
                            mDenoiseModeIndex = scalingMode;
                            mDenoiseModeInitialized = true;
                            m_render->resetTemporalHistory();
                        }
                        ImGui::EndDisabled();
                    }
                }

                ImGui::SeparatorText("Convergence");

                auto maxDepth = m_settingsManager->getAs<uint32_t>("render/pt/depth");
                if (ImGui::SliderInt("Max Depth", (int*)&maxDepth, 1, 16))
                {
                    m_settingsManager->setAs<uint32_t>("render/pt/depth", maxDepth);
                }


                auto sppSubframe = m_settingsManager->getAs<uint32_t>("render/pt/spp");
                if (ImGui::SliderInt("SPP per frame", (int*)&sppSubframe, 1, 32))
                {
                    m_settingsManager->setAs<uint32_t>("render/pt/spp", sppSubframe);
                }
                const bool perFrameDenoise = editor_denoiser::usesPerFrameDenoiseInput(fx, mDenoiseModeIndex) &&
                                             !m_render->denoiserFallbackActive();
                if (perFrameDenoise && ImGui::IsItemHovered())
                {
                    ImGui::SetTooltip("Samples combined into each fresh MetalFX input frame.");
                }

                bool accumulationEnabled = m_settingsManager->getAs<bool>("render/pt/enableAcc");
                const char* accumulationLabel =
                    perFrameDenoise ? "Stop at traced SPP limit" : "Accumulate while still";
                if (ImGui::Checkbox(accumulationLabel, &accumulationEnabled))
                {
                    m_settingsManager->setAs<bool>("render/pt/enableAcc", accumulationEnabled);
                }
                if (perFrameDenoise && ImGui::IsItemHovered())
                {
                    ImGui::SetTooltip(
                        "MetalFX filters each current frame, not the accumulated PT mean.\n"
                        "This limit freezes both tracing and temporal refinement.");
                }

                if (accumulationEnabled)
                {
                    auto sppTotal = m_settingsManager->getAs<uint32_t>("render/pt/sppTotal");
                    const char* limitLabel = perFrameDenoise ? "Traced SPP limit" : "Accumulation SPP limit";
                    if (ImGui::SliderInt(limitLabel, (int*)&sppTotal, 1, 10000))
                    {
                        m_settingsManager->setAs<uint32_t>("render/pt/sppTotal", sppTotal);
                    }
                }
            }

            ImGui::EndTabItem();
        }

        // --- Radiance cache ----------------------------------------------------
        //
        // Everything here changes what the image is, not just how fast it arrives,
        // so every control restarts accumulation. Counting occupancy is a pass over
        // the whole table, so it is asked for only while this node is open -- which
        // is what the setting outside the `if` turns back off again.
        if (ImGui::BeginTabItem("Cache"))
        {
            const bool cachePanelOpen = ImGui::TreeNodeEx("Radiance cache (SHaRC)", ImGuiTreeNodeFlags_DefaultOpen);
            m_settingsManager->setAs<bool>("render/pt/sharcReportOccupancy", cachePanelOpen);
            if (cachePanelOpen)
            {
                auto restart = [this]() { m_sharedCtx->mSubframeIndex = 0; };

                bool cacheEnabled = m_settingsManager->getAs<bool>("render/pt/sharc");
                if (ImGui::Checkbox("Enable", &cacheEnabled))
                {
                    m_settingsManager->setAs<bool>("render/pt/sharc", cacheEnabled);
                    restart();
                }
                if (ImGui::IsItemHovered())
                {
                    ImGui::BeginTooltip();
                    ImGui::TextUnformatted(
                        "Let a path stop after a few bounces and read what the rest of it\n"
                        "would have gathered, averaged over every path that has passed\n"
                        "through the same place. Trades a little bias for path length.");
                    ImGui::EndTooltip();
                }

                ImGui::SameLine();
                if (ImGui::Button("Reset cache"))
                {
                    // Consumed by the renderer on the next frame. Worth having as a
                    // button: the cache deliberately survives camera movement now,
                    // so this is the only way to see a scene cached from nothing.
                    m_settingsManager->setAs<bool>("render/pt/sharcReset", true);
                    restart();
                }

                uint32_t entriesUsed = 0;
                uint32_t capacity = 0;
                if (m_render != nullptr && m_render->radianceCacheOccupancy(entriesUsed, capacity) && capacity > 0)
                {
                    const float occupancy = 100.0f * (float)entriesUsed / (float)capacity;
                    ImGui::Text("Occupancy: %.1f%%  (%u / %u entries)", (double)occupancy, entriesUsed, capacity);
                    // The SDK's own reading of this number, which is the only thing
                    // that makes it actionable.
                    if (occupancy > 60.0f)
                    {
                        ImGui::TextColored(
                            ImVec4(1.0f, 0.6f, 0.0f, 1.0f), "Table is crowded -- raise entries, or evict sooner.");
                    }
                }
                else if (cacheEnabled)
                {
                    ImGui::TextDisabled("Occupancy: not reported by this backend");
                }

                // Entries, as an exponent: the table is masked rather than divided
                // into, so it has to be a power of two, and a free-typed number
                // would only be rounded down behind the user's back.
                const uint32_t entries = m_settingsManager->getAs<uint32_t>("render/pt/sharcCapacity");
                int exponent = 22;
                while ((1u << exponent) > entries && exponent > 16)
                {
                    --exponent;
                }
                if (ImGui::SliderInt("Entries (2^n)", &exponent, 16, 25))
                {
                    m_settingsManager->setAs<uint32_t>("render/pt/sharcCapacity", 1u << exponent);
                    restart();
                }
                if (ImGui::IsItemHovered())
                {
                    ImGui::BeginTooltip();
                    ImGui::Text("%u entries, %.0f MB.\nMore entries means fewer probe runs that come back full.",
                                1u << exponent, (double)(1u << exponent) * 40.0 / 1e6);
                    ImGui::EndTooltip();
                }

                float voxelPixels = m_settingsManager->getAs<float>("render/pt/sharcVoxelPixels");
                if (ImGui::SliderFloat("Voxel size (px)", &voxelPixels, 1.0f, 32.0f, "%.1f"))
                {
                    m_settingsManager->setAs<float>("render/pt/sharcVoxelPixels", voxelPixels);
                    restart();
                }
                if (ImGui::IsItemHovered())
                {
                    ImGui::BeginTooltip();
                    ImGui::TextUnformatted(
                        "How many pixels wide a voxel is in a perspective view. The size follows\n"
                        "the distance to the camera, so one number means the same thing in a room\n"
                        "and in a forest. Use the 'Cache: voxel grid' debug view to set it.");
                    ImGui::EndTooltip();
                }

                uint32_t firstBounce = m_settingsManager->getAs<uint32_t>("render/pt/sharcDepth");
                if (ImGui::SliderInt("First cached bounce", (int*)&firstBounce, 0, 8))
                {
                    m_settingsManager->setAs<uint32_t>("render/pt/sharcDepth", firstBounce);
                    restart();
                }
                if (ImGui::IsItemHovered())
                {
                    ImGui::BeginTooltip();
                    ImGui::TextUnformatted(
                        "Bounces before this are always traced. The camera ray and the first\n"
                        "bounce carry the detail a voxel average would blur, so reading the\n"
                        "cache too early shows up as flat, blotchy indirect light.");
                    ImGui::EndTooltip();
                }

                uint32_t readFrames = m_settingsManager->getAs<uint32_t>("render/pt/sharcReadFrames");
                if (ImGui::SliderInt("Read until (samples)", (int*)&readFrames, 0, 1024))
                {
                    m_settingsManager->setAs<uint32_t>("render/pt/sharcReadFrames", readFrames);
                    restart();
                }
                if (ImGui::IsItemHovered())
                {
                    ImGui::BeginTooltip();
                    ImGui::TextUnformatted(
                        "Stop reading the cache once this many samples have accumulated;\n"
                        "0 never stops. Deposits carry on either way, so the cache is warm\n"
                        "the moment the camera moves again.\n\n"
                        "This is not a taste setting. The cache's error is one value per\n"
                        "voxel held across a temporal window, so it is correlated in space\n"
                        "and time and does not average away -- it is a floor, while plain\n"
                        "path tracing keeps converging past it. Measured on the isometric\n"
                        "bathroom against a 4096-spp reference, structured error only:\n"
                        "  16 spp   0.154 without, 0.098 with   -- cache half the error\n"
                        "  64 spp   0.063 without, 0.039 with   -- cache half the error\n"
                        " 256 spp   0.022 without, 0.019 with   -- level\n"
                        "1024 spp   0.006 without, 0.012 with   -- cache twice the error\n\n"
                        "So the cache is both the faster and the better image while you are\n"
                        "moving, and the thing in the way once you stop.");
                    ImGui::EndTooltip();
                }

                uint32_t minSamples = m_settingsManager->getAs<uint32_t>("render/pt/sharcMinSamples");
                if (ImGui::SliderInt("Min samples to read", (int*)&minSamples, 1, 256))
                {
                    m_settingsManager->setAs<uint32_t>("render/pt/sharcMinSamples", minSamples);
                    restart();
                }
                if (ImGui::IsItemHovered())
                {
                    ImGui::BeginTooltip();
                    ImGui::TextUnformatted(
                        "How much a voxel has to have seen before a path will believe it.\n"
                        "Too low and the cache spreads one path's noise over a region.");
                    ImGui::EndTooltip();
                }

                uint32_t accumFrames = m_settingsManager->getAs<uint32_t>("render/pt/sharcAccumFrames");
                if (ImGui::SliderInt("Temporal window (frames)", (int*)&accumFrames, 1, 256))
                {
                    m_settingsManager->setAs<uint32_t>("render/pt/sharcAccumFrames", accumFrames);
                    restart();
                }
                if (ImGui::IsItemHovered())
                {
                    ImGui::BeginTooltip();
                    ImGui::TextUnformatted(
                        "How many frames a voxel averages over. Larger is quieter and slower\n"
                        "to notice that the lighting changed -- a light switched on takes\n"
                        "roughly this many frames to appear in the cache.");
                    ImGui::EndTooltip();
                }

                uint32_t staleFrames = m_settingsManager->getAs<uint32_t>("render/pt/sharcStaleFrames");
                if (ImGui::SliderInt("Evict after (frames)", (int*)&staleFrames, 8, 512))
                {
                    m_settingsManager->setAs<uint32_t>("render/pt/sharcStaleFrames", staleFrames);
                    restart();
                }
                if (ImGui::IsItemHovered())
                {
                    ImGui::BeginTooltip();
                    ImGui::TextUnformatted(
                        "How long an entry survives with nothing deposited into it. This is\n"
                        "what lets the cache outlive a moving camera instead of being thrown\n"
                        "away by it. Evicting too eagerly costs more in re-insertion than the\n"
                        "slots are worth, so small values are clamped.");
                    ImGui::EndTooltip();
                }

                // Responsive lighting. The controls are shown whether or not the
                // scene has a responsive light, because the answer to "why is this
                // doing nothing" is on the light's own panel and a control that is
                // not there cannot say so.
                ImGui::SeparatorText("Responsive lighting");
                bool responsiveEnabled = m_settingsManager->getAs<bool>("render/pt/sharcResponsiveLighting");
                if (ImGui::Checkbox("Enable##sharcResponsive", &responsiveEnabled))
                {
                    m_settingsManager->setAs<bool>("render/pt/sharcResponsiveLighting", responsiveEnabled);
                    restart();
                }
                if (ImGui::IsItemHovered())
                {
                    ImGui::BeginTooltip();
                    ImGui::TextUnformatted(
                        "Cache lights marked Responsive -- on the light's own panel -- in a\n"
                        "second entry per voxel with a much shorter window, so they can change\n"
                        "faster than the rest of the cache follows. Does nothing unless some\n"
                        "light is marked; turning it off here is how you A/B a scene that has\n"
                        "one.");
                    ImGui::EndTooltip();
                }

                uint32_t responsiveFrames = m_settingsManager->getAs<uint32_t>("render/pt/sharcResponsiveFrames");
                if (ImGui::SliderInt("Responsive window (frames)", (int*)&responsiveFrames, 1, 64))
                {
                    m_settingsManager->setAs<uint32_t>("render/pt/sharcResponsiveFrames", responsiveFrames);
                    restart();
                }
                if (ImGui::IsItemHovered())
                {
                    ImGui::BeginTooltip();
                    ImGui::TextUnformatted(
                        "The window responsive entries average over, and how long they survive\n"
                        "unvisited. Both, because they are the same trade: short enough to\n"
                        "follow the light, long enough not to be noise. Well below the window\n"
                        "above, or there is no point having two.");
                    ImGui::EndTooltip();
                }

                // Metal-only cache internals. The controls above mean the same thing
                // on both backends; what is below is the hash map this backend
                // actually has -- a compact 32-bit key, a sparse update pass and an
                // fp16 resolved half. See docs/sharc-metal.md.
                const bool metalBackend = m_render->denoiserKind() == Render::DenoiserKind::eMetalFx;
                if (metalBackend)
                {
                    ImGui::SeparatorText("Metal");

                    // Must match SHARC_DEBUG_* in ShaderTypes.h, in order.
                    const char* const sharcDebugOptions[] = { "Off",
                                                              "Cached-key colors",
                                                              "Query: hit / miss",
                                                              "Cached sample count",
                                                              "Counters in log",
                                                              "Hash bucket collisions" };
                    uint32_t sharcDebug = m_settingsManager->getAs<uint32_t>("render/pt/sharcDebug");
                    sharcDebug = std::min<uint32_t>(sharcDebug, IM_ARRAYSIZE(sharcDebugOptions) - 1u);
                    if (ImGui::BeginCombo("Hash map diagnostics", sharcDebugOptions[sharcDebug]))
                    {
                        for (uint32_t n = 0; n < IM_ARRAYSIZE(sharcDebugOptions); ++n)
                        {
                            const bool selected = sharcDebug == n;
                            if (ImGui::Selectable(sharcDebugOptions[n], selected) && !selected)
                            {
                                sharcDebug = n;
                                m_settingsManager->setAs<uint32_t>("render/pt/sharcDebug", sharcDebug);
                                restart();
                                m_render->resetTemporalHistory();
                            }
                            if (selected)
                            {
                                ImGui::SetItemDefaultFocus();
                            }
                        }
                        ImGui::EndCombo();
                    }
                    if (ImGui::IsItemHovered())
                    {
                        ImGui::BeginTooltip();
                        ImGui::TextUnformatted(
                            "Alongside the four cache views in Debug view above, which both backends\n"
                            "answer. These are about the map rather than the cache: hit / miss is green\n"
                            "where a query is answered and red where the lookup fails, collisions use\n"
                            "NVIDIA's blue-to-red probe-depth palette, and the counters go to the log\n"
                            "as SHARC stats rather than to the image.");
                        ImGui::EndTooltip();
                    }

                    if (ImGui::TreeNode("Metal cache features"))
                    {
                        bool featureChanged = false;
                        auto featureToggle = [&](const char* label, const char* setting) {
                            bool value = m_settingsManager->getAs<bool>(setting);
                            if (ImGui::Checkbox(label, &value))
                            {
                                m_settingsManager->setAs<bool>(setting, value);
                                featureChanged = true;
                            }
                        };
                        featureToggle("Material demodulation", "render/pt/sharcMaterialDemodulation");
                        featureToggle("Separate emissive", "render/pt/sharcSeparateEmissive");
                        featureToggle("Directional radiance (SH)", "render/pt/sharcDirectional");
                        if (ImGui::IsItemHovered())
                        {
                            ImGui::SetTooltip(
                                "Keeps a bright glossy sample from being reused in an unrelated direction, and pays "
                                "for it with a first-order reconstruction of a signal a cell otherwise stores exactly. "
                                "Off by default, as upstream's SH encoding is.");
                        }
                        featureToggle("Responsive lighting", "render/pt/sharcMetalResponsive");
                        if (ImGui::IsItemHovered())
                        {
                            ImGui::SetTooltip(
                                "Not the same switch as the one above: the compact key has no spare bit for a "
                                "per-light tag, so the companion entries hold the whole lighting signal and come "
                                "out of the configured capacity.");
                        }
                        featureToggle("Cache resampling", "render/pt/sharcCacheResampling");
                        featureToggle("Blend adjacent levels", "render/pt/sharcBlendAdjacentLevels");
                        featureToggle("Fade acceleration", "render/pt/sharcFadeAcceleration");

                        constexpr int kSharcMaxPropagationDepth = 4; // ShaderTypes.h ABI limit.
                        auto propagationDepth = m_settingsManager->getAs<uint32_t>("render/pt/sharcPropagationDepth");
                        if (ImGui::SliderInt("Propagation depth", (int*)&propagationDepth, 1, kSharcMaxPropagationDepth))
                        {
                            m_settingsManager->setAs<uint32_t>("render/pt/sharcPropagationDepth", propagationDepth);
                            featureChanged = true;
                        }
                        auto updateDownscale = m_settingsManager->getAs<uint32_t>("render/pt/sharcUpdateDownscale");
                        if (ImGui::SliderInt("Update block size", (int*)&updateDownscale, 1, 16))
                        {
                            m_settingsManager->setAs<uint32_t>("render/pt/sharcUpdateDownscale", updateDownscale);
                            featureChanged = true;
                        }
                        auto metalMinSamples = m_settingsManager->getAs<uint32_t>("render/pt/sharcMetalMinSamples");
                        if (ImGui::SliderInt("Minimum cached samples", (int*)&metalMinSamples, 1, 64))
                        {
                            m_settingsManager->setAs<uint32_t>("render/pt/sharcMetalMinSamples", metalMinSamples);
                            featureChanged = true;
                        }
                        float receiverRoughness = m_settingsManager->getAs<float>("render/pt/sharcRoughnessThreshold");
                        if (ImGui::SliderFloat("Minimum receiver roughness", &receiverRoughness, 0.0f, 1.0f, "%.2f"))
                        {
                            m_settingsManager->setAs<float>("render/pt/sharcRoughnessThreshold", receiverRoughness);
                            featureChanged = true;
                        }
                        if (ImGui::IsItemHovered())
                        {
                            ImGui::SetTooltip(
                                "A receiver with any sharper reflective layer keeps tracing because the current "
                                "cache cannot represent it. Transmission and fibres always keep tracing.");
                        }
                        float radianceScale = m_settingsManager->getAs<float>("render/pt/sharcRadianceScale");
                        if (ImGui::DragFloat("Radiance fixed-point scale", &radianceScale, 10.0f, 1.0f, 100000.0f, "%.0f"))
                        {
                            m_settingsManager->setAs<float>("render/pt/sharcRadianceScale", radianceScale);
                            featureChanged = true;
                        }
                        if (ImGui::IsItemHovered())
                        {
                            ImGui::SetTooltip(
                                "Start at 1000. Reduce it if the counters report 31-32 occupied radiance bits or "
                                "accumulation clamps; a lower value trades fixed-point precision for headroom.");
                        }
                        if (featureChanged)
                        {
                            restart();
                            m_render->resetTemporalHistory();
                        }
                        ImGui::TreePop();
                    }
                }

                ImGui::TreePop();
            }

            ImGui::EndTabItem();
        }

        if (ImGui::BeginTabItem("Output"))
        {
            if (ImGui::Button("Save Preview Screenshot"))
            {
                // Generate default filename with timestamp
                const std::time_t now = std::time(nullptr);
                std::tm localTime{};
                const std::tm* tm = localtime_r(&now, &localTime);
                // localtime() returns null for a clock it cannot convert, and strftime()
                // returns 0 when the result would not fit; either way the dialog still
                // needs a name to open with.
                std::string defaultName = "screenshot.exr";
                char stamp[64];
                if (tm != nullptr && std::strftime(stamp, sizeof(stamp), "screenshot_%Y%m%d_%H%M%S.exr", tm) != 0)
                {
                    defaultName = stamp;
                }

                IGFD::FileDialogConfig config{};
                config.path = ".";
                config.fileName = defaultName;
                ImGuiFileDialog::Instance()->OpenDialog("SaveScreenshotDlgKey", "Save Screenshot", ".exr,.png", config);
            }
            if (ImGui::Checkbox("EXR holds the display image", &m_screenshotDisplayReferred))
            {
                // Nothing to invalidate: the flag is read when the file is written.
            }
            if (ImGui::IsItemHovered())
            {
                ImGui::SetTooltip(
                    "Off: the EXR holds scene-linear radiance, which is what a reference or a\n"
                    "comparison against another renderer wants.\n\n"
                    "On: it holds what the screen shows -- exposure, tone curve and display\n"
                    "headroom applied -- with the range above white intact, which no PNG can\n"
                    "carry. Neither is transfer encoded; a PNG always gets the SDR rendition.");
            }

            ImGui::EndTabItem();
        }

        ImGui::EndTabBar();
    }

    ImGui::End();
}

} // namespace oka
