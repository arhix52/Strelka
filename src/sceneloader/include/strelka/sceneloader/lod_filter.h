#pragma once

#include <cctype>
#include <string>

namespace oka
{

inline bool isProxyOrLowerLod(const std::string& name)
{
    std::string lower(name.size(), '\0');
    for (size_t i = 0; i < name.size(); ++i)
    {
        lower[i] = static_cast<char>(std::tolower(static_cast<unsigned char>(name[i])));
    }

    if (lower.find("proxy") != std::string::npos)
    {
        return true;
    }

    for (size_t pos = lower.find("lod"); pos != std::string::npos; pos = lower.find("lod", pos + 3))
    {
        size_t digit = pos + 3;
        if (digit < lower.size() && lower[digit] == '_')
        {
            ++digit;
        }
        if (digit >= lower.size() || std::isdigit(static_cast<unsigned char>(lower[digit])) == 0)
        {
            continue;
        }
        size_t end = digit;
        while (end < lower.size() && std::isdigit(static_cast<unsigned char>(lower[end])) != 0)
        {
            ++end;
        }
        // Compared as digits rather than parsed: the level is only ever "zero or
        // not", and a name carrying more digits than an integer holds should not
        // decide it by overflowing.
        if (lower.substr(digit, end - digit).find_first_not_of('0') != std::string::npos)
        {
            return true;
        }
    }
    return false;
}

} // namespace oka
