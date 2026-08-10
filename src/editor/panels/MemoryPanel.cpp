#include "../EditorApp.h"

#include "imgui.h"

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace oka
{
namespace
{

// A palette rather than a hue ramp: adjacent slices in a pie have to be told
// apart at a glance, and evenly spaced hues put two greens next to each other as
// soon as there are more than about six categories.
const ImU32 kSliceColors[] = {
    IM_COL32(0x4e, 0x9a, 0xe0, 0xff), IM_COL32(0xe0, 0x7b, 0x39, 0xff), IM_COL32(0x62, 0xb5, 0x62, 0xff),
    IM_COL32(0xd0, 0x5c, 0x5c, 0xff), IM_COL32(0xa9, 0x7c, 0xd8, 0xff), IM_COL32(0xb0, 0x83, 0x60, 0xff),
    IM_COL32(0xe0, 0x8e, 0xc4, 0xff), IM_COL32(0x9a, 0x9a, 0x9a, 0xff), IM_COL32(0xc8, 0xc8, 0x4a, 0xff),
    IM_COL32(0x4a, 0xc0, 0xc8, 0xff), IM_COL32(0x86, 0x6f, 0xb0, 0xff), IM_COL32(0x6f, 0x8f, 0x4a, 0xff),
    IM_COL32(0xd8, 0xa0, 0x50, 0xff), IM_COL32(0x50, 0x78, 0xa8, 0xff), IM_COL32(0xa8, 0x50, 0x78, 0xff),
};

std::string humanBytes(size_t bytes)
{
    const double b = (double)bytes;
    if (bytes >= 1024ull * 1024 * 1024)
        return fmt::format("{:.2f} GB", b / (1024.0 * 1024.0 * 1024.0));
    if (bytes >= 1024ull * 1024)
        return fmt::format("{:.0f} MB", b / (1024.0 * 1024.0));
    return fmt::format("{:.0f} KB", b / 1024.0);
}

// Drawn here rather than pulled in with ImPlot. ImPlot has PlotPieChart and is
// the usual answer, but its released recipes target ImGui 1.90 and this project
// is on 1.92 with the new texture API -- a dependency that may or may not build,
// for something the draw list does in twenty lines.
void drawPie(const std::vector<Render::MemoryReport::Entry>& entries, size_t total, float radius)
{
    ImDrawList* draw = ImGui::GetWindowDrawList();
    const ImVec2 topLeft = ImGui::GetCursorScreenPos();
    const ImVec2 centre(topLeft.x + radius, topLeft.y + radius);

    float angle = -IM_PI * 0.5f; // start at twelve o'clock
    for (size_t i = 0; i < entries.size() && total > 0; ++i)
    {
        const float sweep = 2.0f * IM_PI * (float)((double)entries[i].bytes / (double)total);
        // Under about a degree the arc degenerates and the fill draws nothing,
        // which would silently drop the slice. Skipping it explicitly at least
        // keeps the colours aligned with the legend.
        if (sweep > 0.005f)
        {
            const int segments = std::max(3, (int)(sweep * 24.0f));
            draw->PathLineTo(centre);
            draw->PathArcTo(centre, radius, angle, angle + sweep, segments);
            draw->PathFillConvex(kSliceColors[i % IM_ARRAYSIZE(kSliceColors)]);
        }
        angle += sweep;
    }
    ImGui::Dummy(ImVec2(radius * 2.0f, radius * 2.0f));
}

void drawBreakdown(const char* title,
                   std::vector<Render::MemoryReport::Entry> entries,
                   size_t groundTruth,
                   const char* groundTruthLabel)
{
    // Largest first: the question this panel answers is always "what is the big
    // one", and a fixed declaration order buries it.
    std::sort(entries.begin(), entries.end(),
              [](const Render::MemoryReport::Entry& a, const Render::MemoryReport::Entry& b) {
                  return a.bytes > b.bytes;
              });

    size_t accounted = 0;
    for (const auto& e : entries)
    {
        accounted += e.bytes;
    }
    // What the categories do not explain, shown rather than dropped. A buffer
    // added to the renderer and not to the report lands here instead of making
    // the chart quietly lie about proportions.
    if (groundTruth > accounted)
    {
        entries.push_back({ "Unaccounted", groundTruth - accounted });
    }
    const size_t total = std::max(accounted, groundTruth);

    ImGui::SeparatorText(title);
    ImGui::TextDisabled("%s: %s", groundTruthLabel, humanBytes(groundTruth).c_str());

    const float radius = 70.0f;
    drawPie(entries, total, radius);
    ImGui::SameLine();

    ImGui::BeginGroup();
    for (size_t i = 0; i < entries.size(); ++i)
    {
        const ImU32 colour = kSliceColors[i % IM_ARRAYSIZE(kSliceColors)];
        ImDrawList* draw = ImGui::GetWindowDrawList();
        const ImVec2 p = ImGui::GetCursorScreenPos();
        const float h = ImGui::GetTextLineHeight();
        draw->AddRectFilled(ImVec2(p.x, p.y + 2.0f), ImVec2(p.x + h - 4.0f, p.y + h - 2.0f), colour);
        ImGui::Dummy(ImVec2(h, h));
        ImGui::SameLine();
        ImGui::Text("%-22s %9s  %4.1f%%", entries[i].name, humanBytes(entries[i].bytes).c_str(),
                    total > 0 ? 100.0 * (double)entries[i].bytes / (double)total : 0.0);
    }
    ImGui::EndGroup();
}

} // namespace

void EditorApp::drawMemoryPanel()
{
    ImGui::Begin("Memory:", &m_showMemory);

    Render::MemoryReport report;
    if (!m_render || !m_render->memoryReport(report))
    {
        ImGui::TextDisabled("This backend does not report memory.");
        ImGui::End();
        return;
    }

    drawBreakdown("GPU", report.gpu, report.deviceAllocated, "device allocated");

    ImGui::Spacing();
    // On unified memory the process footprint already contains everything the
    // device allocated, so the CPU side is that minus the GPU rather than a
    // separate pool -- stating it any other way would double the total.
    std::vector<Render::MemoryReport::Entry> cpu = report.cpu;
    cpu.push_back({ "GPU (in footprint)", report.deviceAllocated });
    drawBreakdown("Process", cpu, report.processFootprint, "phys_footprint");

    ImGui::End();
}

} // namespace oka
