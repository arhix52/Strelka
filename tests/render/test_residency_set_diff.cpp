#include <doctest/doctest.h>

#include "residency_set_diff.h"

#include <algorithm>
#include <cstdint>
#include <unordered_set>

using oka::metal::retiredResidencyAllocations;

TEST_CASE("residency diff retires replaced allocations")
{
    const std::unordered_set<uintptr_t> previous = { 1, 2, 3, 4 };
    const std::unordered_set<uintptr_t> current = { 2, 4, 5 };

    std::vector<uintptr_t> retired = retiredResidencyAllocations(previous, current);
    std::ranges::sort(retired);

    REQUIRE(retired.size() == 2);
    CHECK(retired[0] == 1);
    CHECK(retired[1] == 3);
}

TEST_CASE("residency diff keeps an unchanged generation")
{
    const std::unordered_set<uintptr_t> allocations = { 10, 20 };

    CHECK(retiredResidencyAllocations(allocations, allocations).empty());
}
