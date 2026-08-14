#pragma once

#include <log.h>
#include <spdlog/fmt/fmt.h>
#include <functional>
#include <map>
#include <string>
#include <string_view>
#include <variant>
#include <cassert>

namespace oka
{

class SettingsManager
{
private:
    using SettingValue = std::variant<uint32_t, float, bool, std::string>;

    // std::less<> is a transparent comparator, so find() accepts a std::string_view
    // (and thus a raw `const char*`) directly. std::unordered_map only gained
    // heterogeneous lookup in C++20 — with the
    // hash map every get/set had to materialise a std::string key, which heap
    // allocates for any key longer than the SSO buffer (e.g.
    // "render/post/tonemapper/shutterSpeed"). The UI issues dozens of those per
    // frame; the settings table only holds a few dozen entries, so the O(log n)
    // tree lookup is cheaper than the allocation it replaces.
    std::map<std::string, SettingValue, std::less<>> mMap;

public:
    SettingsManager(/* args */) = default;
    ~SettingsManager() = default;

    template <typename T>
    void setAs(std::string_view name, const T& value)
    {
        if (auto it = mMap.find(name); it != mMap.end())
        {
            it->second = value;
            return;
        }
        mMap.emplace(std::string(name), SettingValue(value));
    }

    /// Read a setting. A missing key or a type mismatch is reported and yields a
    /// default-constructed value rather than inserting a bogus entry (the old
    /// `mMap[name]` did) or throwing std::bad_variant_access from the render loop.
    template <typename T>
    T getAs(std::string_view name) const
    {
        const auto it = mMap.find(name);
        if (it == mMap.end())
        {
            STRELKA_ERROR("The setting {} does not exist", name);
            assert(0);
            return T{};
        }
        if (const T* value = std::get_if<T>(&it->second))
        {
            return *value;
        }
        STRELKA_ERROR("The setting {} is stored with a different type", name);
        assert(0);
        return T{};
    }

    bool contains(std::string_view name) const
    {
        return mMap.find(name) != mMap.end();
    }

    void erase(std::string_view name)
    {
        if (const auto it = mMap.find(name); it != mMap.end())
        {
            mMap.erase(it);
        }
    }

    /// Erase all keys starting with the given prefix.
    /// Keys are sorted, so the matching range is contiguous: seek to it instead
    /// of scanning the whole table.
    void eraseByPrefix(std::string_view p)
    {
        for (auto it = mMap.lower_bound(p); it != mMap.end();)
        {
            if (std::string_view(it->first).substr(0, p.size()) != p)
                break;
            it = mMap.erase(it);
        }
    }
};

/// The per-animation settings keys.
///
/// A dozen call sites -- the editor, the animation panel, the Metal renderer --
/// build these, and getAs() treats a key that differs by one character as a
/// missing setting: it logs, asserts, and hands back a default. Spelling them
/// once is what keeps a writer and its reader on the same key.
inline std::string animationStateKey(size_t index)
{
    return fmt::format("render/animation/anim{}/state", index);
}

inline std::string animationTimeKey(size_t index)
{
    return fmt::format("render/animation/anim{}/time", index);
}

} // namespace oka
