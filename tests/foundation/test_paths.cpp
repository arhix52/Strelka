#include <doctest/doctest.h>

#include <application_paths.h>

TEST_CASE("application writable directories are distinct and product-scoped")
{
    const std::filesystem::path support = oka::applicationSupportDirectory();
    const std::filesystem::path cache = oka::applicationCacheDirectory();
    const std::filesystem::path logs = oka::applicationLogDirectory();

    CHECK_FALSE(support.empty());
    CHECK_FALSE(cache.empty());
    CHECK_FALSE(logs.empty());
    CHECK(support != cache);
    CHECK(support != logs);
    CHECK(cache != logs);
#if defined(__APPLE__) || defined(_WIN32)
    CHECK(support.filename() == "Strelka");
    CHECK(cache.filename() == "Strelka");
    CHECK(logs.filename() == "Strelka");
#else
    CHECK(support.filename() == "strelka");
    CHECK(cache.filename() == "strelka");
    CHECK(logs.filename() == "strelka");
#endif
}
