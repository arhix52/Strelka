#pragma once

#include <log.h>
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
            STRELKA_ERROR("The setting {} does not exist", name);
            assert(0);
        }
    }

public:
    SettingsManager(/* args */) = default;
    ~SettingsManager() = default;

    template <typename T>
    void setAs(const char* name, const T& value)
    {
        // mMap[name] = std::to_string(value);
        mMap[name] = toString(value);
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

    static std::string toString(const std::string& value)
    {
        return value;
    }

    template <typename T>
    std::string toString(const T& value)
    {
        // Default implementation for non-specialized types
        return std::to_string(value);
    }
};

template<>
inline void SettingsManager::setAs(const char* name, const std::string& value)
{
    mMap[name] = value;
}

template <>
inline bool SettingsManager::convertValue(const std::string& value)
{
    return static_cast<bool>(atoi(value.c_str()));
}

template <>
inline float SettingsManager::convertValue(const std::string& value)
{
    return static_cast<float>(atof(value.c_str()));
}

template <>
inline uint32_t SettingsManager::convertValue(const std::string& value)
{
    return static_cast<uint32_t>(atoi(value.c_str()));
}

template <>
inline std::string SettingsManager::convertValue(const std::string& value)
{
    return value;
}

template <>
inline std::string SettingsManager::toString(const std::string& value)
{
    return value;
}

} // namespace oka
