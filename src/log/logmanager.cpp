#include "logmanager.h"

#include <spdlog/spdlog.h>
#include <spdlog/cfg/env.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <memory>
#include <vector>

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
        auto fileSink = std::make_shared<spdlog::sinks::basic_file_sink_mt>("strelka.log");
        std::vector<spdlog::sink_ptr> sinks = { consolesink, fileSink };
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
