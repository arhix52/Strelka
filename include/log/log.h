#pragma once

#include <spdlog/spdlog.h>

#define STRELKA_DEFAULT_LOGGER_NAME "Strelka"

#ifndef RELEASE
#define STRELKA_TRACE(...) if (spdlog::get(STRELKA_DEFAULT_LOGGER_NAME) != nullptr) {spdlog::get(STRELKA_DEFAULT_LOGGER_NAME)->trace(__VA_ARGS__);}
#define STRELKA_DEBUG(...) if (spdlog::get(STRELKA_DEFAULT_LOGGER_NAME) != nullptr) {spdlog::get(STRELKA_DEFAULT_LOGGER_NAME)->debug(__VA_ARGS__);}
#define STRELKA_INFO(...) if (spdlog::get(STRELKA_DEFAULT_LOGGER_NAME) != nullptr) {spdlog::get(STRELKA_DEFAULT_LOGGER_NAME)->info(__VA_ARGS__);}
#define STRELKA_WARNING(...) if (spdlog::get(STRELKA_DEFAULT_LOGGER_NAME) != nullptr) {spdlog::get(STRELKA_DEFAULT_LOGGER_NAME)->warn(__VA_ARGS__);}
#define STRELKA_ERROR(...) if (spdlog::get(STRELKA_DEFAULT_LOGGER_NAME) != nullptr) {spdlog::get(STRELKA_DEFAULT_LOGGER_NAME)->error(__VA_ARGS__);}
#define STRELKA_FATAL(...) if (spdlog::get(STRELKA_DEFAULT_LOGGER_NAME) != nullptr) {spdlog::get(STRELKA_DEFAULT_LOGGER_NAME)->critical(__VA_ARGS__);}
#else
define STRELKA_TRACE(...) void(0);
define STRELKA_DEBUG(...) void(0);
define STRELKA_INFO(...) void(0);
define STRELKA_WARNING(...) void(0);
define STRELKA_ERROR(...) void(0);
define STRELKA_FATAL(...) void(0);
#endif
