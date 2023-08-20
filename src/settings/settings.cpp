#include "settings/settings.h"

namespace oka
{

template<>
void SettingsManager::setAs(const char* name, const std::string& value)
{
    mMap[name] = value;
}

template <>
bool SettingsManager::convertValue(const std::string& value)
{
    return static_cast<bool>(atoi(value.c_str()));
}

template <>
float SettingsManager::convertValue(const std::string& value)
{
    return static_cast<float>(atof(value.c_str()));
}

template <>
uint32_t SettingsManager::convertValue(const std::string& value)
{
    return static_cast<uint32_t>(atoi(value.c_str()));
}

template <>
std::string SettingsManager::toString(const std::string& value)
{
    return value;
}

} // namespace oka
