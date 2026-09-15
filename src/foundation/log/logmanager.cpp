#include "logmanager.h"

#include <spdlog/spdlog.h>
#include <spdlog/cfg/env.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <vector>

namespace
{

/// Where strelka.log may go, best first.
std::vector<std::filesystem::path> logFileCandidates()
{
    std::vector<std::filesystem::path> candidates;
    candidates.emplace_back("strelka.log");

    // NOLINTNEXTLINE(concurrency-mt-unsafe)
    if (const char* stateHome = std::getenv("XDG_STATE_HOME"); stateHome != nullptr && *stateHome != '\0')
    {
        candidates.emplace_back(std::filesystem::path(stateHome) / "strelka" / "strelka.log");
    }
    // NOLINTNEXTLINE(concurrency-mt-unsafe)
    else if (const char* home = std::getenv("HOME"); home != nullptr && *home != '\0')
    {
        candidates.emplace_back(std::filesystem::path(home) / ".local" / "state" / "strelka" / "strelka.log");
    }
    return candidates;
}

} // namespace

oka::Logmanager::Logmanager()
{
    initialize();
}

oka::Logmanager::~Logmanager()
{
    shutdown();
}

void oka::Logmanager::initialize()
{
    auto logger = spdlog::get("Strelka");
    if (!logger)
    {
        spdlog::cfg::load_env_levels();
        auto consolesink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
        std::vector<spdlog::sink_ptr> sinks = { consolesink };

        for (const std::filesystem::path& candidate : logFileCandidates())
        {
            try
            {
                std::error_code ec;
                if (candidate.has_parent_path())
                {
                    std::filesystem::create_directories(candidate.parent_path(), ec);
                }
                sinks.push_back(std::make_shared<spdlog::sinks::basic_file_sink_mt>(candidate.string()));
                break;
            }
            // NOLINTNEXTLINE(bugprone-empty-catch)
            catch (const spdlog::spdlog_ex&)
            {
                // Next candidate; the console sink already holds the session.
            }
        }

        logger = std::make_shared<spdlog::logger>("Strelka", sinks.begin(), sinks.end());

        // TODO: env var doesn't work on linux
        logger->set_level(spdlog::level::trace);
        logger->flush_on(spdlog::level::trace);

#if defined(WIN32)
        logger->set_level(spdlog::level::trace);
        logger->flush_on(spdlog::level::trace);
#endif
        spdlog::register_logger(logger);
    }
}

void oka::Logmanager::shutdown()
{
    spdlog::shutdown();
}
