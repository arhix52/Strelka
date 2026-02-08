#pragma once

#include <spdlog/spdlog.h>

namespace oka
{
class Logmanager
{
public:
    Logmanager(/* args */);
    ~Logmanager();

    void initialize();
    void shutdown();
};

} // namespace oka
