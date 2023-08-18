#pragma once

#include <iostream>
#include <string>
#include <unordered_map>
#include <cassert>

namespace oka
{

class SettingsManager
{
private:
    /* data */
    std::unordered_map<std::string, std::string> mMap;

    void isNameValid(const char* name)
    {
        if (mMap.find(name) == mMap.end())
        {
            std::cerr << "The setting " << name << " does not exist" << std::endl;
            assert(0);
        }
    }

public:
    SettingsManager(/* args */) = default;
    ~SettingsManager() = default;

    template <typename T>
    void setAs(const char* name, const T& value)
    {
        mMap[name] = std::to_string(value);
    }

    void setAs(const char* name, const std::string& value)
    {
        mMap[name] = value;
    }

    template <typename T>
    T getAs(const char* name)
    {
        isNameValid(name);
        return convertValue<T>(mMap[name]);
    }

private:
    template <typename T>
    T convertValue(const std::string& value)
    {
        // Default implementation for non-specialized types
        return T{};
    }
};

// Explicit template specializations in the namespace scope
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

} // namespace oka
