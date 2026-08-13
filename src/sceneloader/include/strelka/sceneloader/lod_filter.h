#pragma once

#include <cctype>
#include <string>

namespace oka
{

// Which of a set of alternative representations of the same object to draw.
//
// glTF can say this properly -- MSFT_lod -- but scenes exported from Blender do
// not, and encode it in node names instead: cover_01_lod0 beside cover_01_lod1,
// rock_proxy beside the detailed rock. A loader that takes every node at face
// value draws them all, stacked in the same space, and because the alternatives
// are near-coincident, which surface a ray reaches first is decided by numerical
// accident.
//
// Kept in a header rather than inside the loader so the rule can be tested
// directly; it is a string convention, and string conventions go wrong at the
// edges (lod10 is not lod1, LOD_0 is still level zero, "lodge" is a word).
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

    // Level zero is the one to keep; any level above it is an alternative to
    // geometry that is already being drawn. A "lod" that is not followed by a
    // number names nothing in particular -- it is part of a word -- and is left
    // alone.
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
