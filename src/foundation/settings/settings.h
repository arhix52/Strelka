#pragma once

#include <log.h>
#include <string>
#include <unordered_map>
#include <variant>
#include <cassert>

namespace oka
{

class SettingsManager
{
private:
    using SettingValue = std::variant<uint32_t, float, bool, std::string>;
    std::unordered_map<std::string, SettingValue> mMap;

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
        mMap[name] = value;
    }

    template <typename T>
    T getAs(const char* name)
    {
        isNameValid(name);
        return std::get<T>(mMap[name]);
    }
};

} // namespace oka
