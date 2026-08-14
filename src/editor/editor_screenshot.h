#pragma once

#include <string_view>

namespace oka
{
namespace editor_screenshot
{

enum class Source
{
    LinearPreview,
    DisplayPreview,
};

inline Source sourceForExtension(std::string_view extension)
{
    return extension == ".png" ? Source::DisplayPreview : Source::LinearPreview;
}

} // namespace editor_screenshot
} // namespace oka
