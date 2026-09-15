#pragma once

#include <filesystem>

namespace oka
{

std::filesystem::path applicationSupportDirectory();
std::filesystem::path applicationCacheDirectory();
std::filesystem::path applicationLogDirectory();

} // namespace oka
