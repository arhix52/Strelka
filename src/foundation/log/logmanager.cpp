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

    // $XDG_STATE_HOME, or the default the spec gives for it. State is the right
    // category for a log: not configuration, not a cache, and not something the
    // user opens.
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

        // The working directory first, because that is where every harness and
        // every note in the docs looks for strelka.log, and where a developer
        // running from build/Release expects it.
        //
        // It is not always writable. Launched from a desktop entry the working
        // directory is whatever the session manager had -- "/" here -- and the
        // constructor below throws. That threw out of Logmanager's constructor,
        // out of main, and terminated: the editor could not be started from its
        // own menu entry at all, with "Failed opening file strelka.log for
        // writing: Permission denied" on a console nobody sees.
        //
        // So: the working directory if it takes the file, the XDG state
        // directory if it does not, and the console alone if neither does. A log
        // file is a convenience; refusing to start is not a proportionate answer
        // to not having one.
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
