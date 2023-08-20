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

    std::string toString(const std::string& value)
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

} // namespace oka
