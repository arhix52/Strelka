#pragma once

#include <unordered_set>
#include <vector>


namespace oka::metal
{

template <typename T>
std::vector<T> retiredResidencyAllocations(const std::unordered_set<T>& previous,
                                           const std::unordered_set<T>& current)
{
    std::vector<T> retired;
    retired.reserve(previous.size());
    for (const T allocation : previous)
    {
        if (current.find(allocation) == current.end())
        {
            retired.push_back(allocation);
        }
    }
    return retired;
}

} // namespace oka::metal

