#include <strelka/display/output_probe.h>

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <string>
#include <vector>

#if defined(__linux__)
#include <chrono>
#include <csignal>
#include <fcntl.h>
#include <poll.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#elif defined(_WIN32)
#include <windows.h>
#endif

namespace oka
{
namespace display_output
{
namespace
{
struct GdctlMode
{
    float refreshRateHz = 0.0f;
    bool current = false;
    bool variable = false;
};

struct GdctlMonitor
{
    std::string connector;
    std::string displayName;
    std::string productName;
    std::vector<GdctlMode> modes;
    float minRefreshRateHz = 0.0f;
};

std::string trim(std::string_view value)
{
    size_t first = 0;
    size_t last = value.size();

    while (first < last &&
           std::isspace(static_cast<unsigned char>(value[first])) != 0)
    {
        ++first;
    }
    while (last > first &&
           std::isspace(static_cast<unsigned char>(value[last - 1])) != 0)
    {
        --last;
    }
    return std::string(value.substr(first, last - first));
}

std::string normalizedName(std::string_view value)
{
    std::string result;
    size_t index = 0;
    unsigned char character = 0;

    result.reserve(value.size());
    while (index < value.size())
    {
        character = static_cast<unsigned char>(value[index]);
        if (std::isalnum(character) != 0)
        {
            result.push_back(static_cast<char>(std::tolower(character)));
        }
        ++index;
    }
    return result;
}

bool parsePositiveFloat(std::string_view value, float *result)
{
    std::string text;
    // std::strtof writes back through &end, so it must stay a mutable char*.
    // NOLINTNEXTLINE(misc-const-correctness)
    char *end = nullptr;
    float parsed = 0.0f;

    if (result == nullptr)
    {
        return false;
    }
    text = trim(value);
    errno = 0;
    parsed = std::strtof(text.c_str(), &end);
    if (errno != 0 || end == text.c_str() || !std::isfinite(parsed) ||
        parsed <= 0.0f)
    {
        return false;
    }
    *result = parsed;
    return true;
}

std::string valueAfter(std::string_view line, std::string_view marker)
{
    size_t markerPosition = 0;
    size_t arrowPosition = 0;

    markerPosition = line.find(marker);
    if (markerPosition == std::string_view::npos)
    {
        return {};
    }
    arrowPosition = line.find("\xE2\x87\x92", markerPosition + marker.size());
    if (arrowPosition != std::string_view::npos)
    {
        return trim(line.substr(arrowPosition + 3));
    }
    return trim(line.substr(markerPosition + marker.size()));
}

int monitorMatchScore(const GdctlMonitor& monitor,
                      std::string_view requestedName)
{
    std::string requested;
    std::string connector;
    std::string displayName;
    std::string productName;

    requested = normalizedName(requestedName);
    connector = normalizedName(monitor.connector);
    displayName = normalizedName(monitor.displayName);
    productName = normalizedName(monitor.productName);
    if (requested.empty())
    {
        return 0;
    }
    if (requested == displayName || requested == productName ||
        requested == connector)
    {
        return 3;
    }
    if ((!displayName.empty() &&
         (requested.find(displayName) != std::string::npos ||
          displayName.find(requested) != std::string::npos)) ||
        (!connector.empty() &&
         (requested.find(connector) != std::string::npos ||
          connector.find(requested) != std::string::npos)) ||
        (!productName.empty() &&
         (requested.find(productName) != std::string::npos ||
          productName.find(requested) != std::string::npos)))
    {
        return 2;
    }
    return 0;
}

PlatformDisplayState stateFromMonitor(const GdctlMonitor& monitor)
{
    PlatformDisplayState result;
    const GdctlMode *currentMode = nullptr;
    bool supportsVariable = false;
    size_t index = 0;

    while (index < monitor.modes.size())
    {
        supportsVariable = supportsVariable || monitor.modes[index].variable;
        if (monitor.modes[index].current)
        {
            currentMode = &monitor.modes[index];
        }
        if (monitor.modes[index].variable)
        {
            result.maxRefreshRateHz =
                std::max(result.maxRefreshRateHz,
                         monitor.modes[index].refreshRateHz);
        }
        ++index;
    }

    result.minRefreshRateHz = monitor.minRefreshRateHz;
    if (currentMode != nullptr)
    {
        result.currentRefreshRateHz = currentMode->refreshRateHz;
    }
    if (!supportsVariable && result.minRefreshRateHz <= 0.0f)
    {
        result.vrrStatus = VrrStatus::Unknown;
        return result;
    }
    if (currentMode == nullptr)
    {
        result.vrrStatus = VrrStatus::Supported;
        return result;
    }

    // The current mode's refresh is the upper edge of its VRR range. Do not use
    // a similarly clocked mode from another resolution as this monitor's bound.
    result.maxRefreshRateHz = currentMode->refreshRateHz;
    result.vrrStatus =
        currentMode->variable ? VrrStatus::Active : VrrStatus::Fixed;
    return result;
}

#if defined(__linux__)
PlatformDisplayState probeGdctl(std::string_view monitorName)
{
    constexpr size_t kMaximumOutputBytes = size_t{ 1024 } * 1024;
    constexpr std::chrono::milliseconds kTimeout(1500);
    int pipeDescriptors[2] = {-1, -1};
    pid_t child = -1;
    int descriptorFlags = 0;
    int childStatus = 0;
    pid_t waitResult = 0;
    std::string output;
    std::chrono::steady_clock::time_point deadline;
    bool childDone = false;
    bool outputClosed = false;
    bool failed = false;
    char buffer[4096];
    ssize_t bytesRead = 0;
    pollfd descriptor = {};
    int pollResult = 0;
    int nullDescriptor = -1;

    // NOLINTNEXTLINE(concurrency-mt-unsafe)
    if (std::getenv("WAYLAND_DISPLAY") == nullptr ||
        access("/usr/bin/gdctl", X_OK) != 0 || pipe(pipeDescriptors) != 0)
    {
        return {};
    }

    child = fork();
    if (child == 0)
    {
        close(pipeDescriptors[0]);
        dup2(pipeDescriptors[1], STDOUT_FILENO);
        close(pipeDescriptors[1]);
        nullDescriptor = open("/dev/null", O_WRONLY);
        if (nullDescriptor >= 0)
        {
            dup2(nullDescriptor, STDERR_FILENO);
            close(nullDescriptor);
        }
        // NOLINTNEXTLINE(concurrency-mt-unsafe)
        setenv("PATH", "/usr/bin:/bin", 1);
        execl("/usr/bin/gdctl", "gdctl", "show", "--modes",
              "--properties", static_cast<char *>(nullptr));
        _exit(127);
    }
    close(pipeDescriptors[1]);
    pipeDescriptors[1] = -1;
    if (child < 0)
    {
        close(pipeDescriptors[0]);
        return {};
    }

    descriptorFlags = fcntl(pipeDescriptors[0], F_GETFL, 0);
    if (descriptorFlags >= 0)
    {
        fcntl(pipeDescriptors[0], F_SETFL, descriptorFlags | O_NONBLOCK);
    }
    output.reserve(16384);
    deadline = std::chrono::steady_clock::now() + kTimeout;
    descriptor.fd = pipeDescriptors[0];
    descriptor.events = POLLIN | POLLHUP;
    while (!childDone || !outputClosed)
    {
        while (true)
        {
            bytesRead = read(pipeDescriptors[0], buffer, sizeof(buffer));
            if (bytesRead <= 0)
            {
                break;
            }
            if (output.size() + static_cast<size_t>(bytesRead) >
                kMaximumOutputBytes)
            {
                failed = true;
                break;
            }
            output.append(buffer, static_cast<size_t>(bytesRead));
        }
        if (bytesRead == 0)
        {
            outputClosed = true;
        }
        if (failed)
        {
            break;
        }

        waitResult = waitpid(child, &childStatus, WNOHANG);
        if (waitResult == child)
        {
            childDone = true;
        }
        else if (waitResult < 0 && errno != EINTR)
        {
            failed = true;
            break;
        }
        if (childDone && outputClosed)
        {
            break;
        }
        if (std::chrono::steady_clock::now() >= deadline)
        {
            failed = true;
            break;
        }
        pollResult = poll(&descriptor, 1, 25);
        if (pollResult < 0 && errno != EINTR)
        {
            failed = true;
            break;
        }
    }

    close(pipeDescriptors[0]);
    if (!childDone)
    {
        kill(child, SIGKILL);
        while (waitpid(child, &childStatus, 0) < 0 && errno == EINTR)
        {
        }
    }
    if (failed || !WIFEXITED(childStatus) || WEXITSTATUS(childStatus) != 0)
    {
        return {};
    }
    return parseGdctlOutput(output, monitorName);
}
#endif
} // namespace

PlatformDisplayState parseGdctlOutput(std::string_view output,
                                     std::string_view monitorName)
{
    std::vector<GdctlMonitor> monitors;
    GdctlMonitor *monitor = nullptr;
    GdctlMode *mode = nullptr;
    size_t lineStart = 0;
    size_t lineEnd = 0;
    std::string_view line;
    size_t markerPosition = 0;
    std::string header;
    size_t connectorEnd = 0;
    float refreshRate = 0.0f;
    int bestScore = -1;
    const GdctlMonitor *bestMonitor = nullptr;
    size_t index = 0;
    int score = 0;

    while (lineStart < output.size())
    {
        lineEnd = output.find('\n', lineStart);
        if (lineEnd == std::string_view::npos)
        {
            lineEnd = output.size();
        }
        line = output.substr(lineStart, lineEnd - lineStart);
        if (line.find("Logical monitors:") != std::string_view::npos)
        {
            break;
        }

        markerPosition = line.find("Monitor ");
        if (markerPosition != std::string_view::npos)
        {
            monitors.emplace_back();
            monitor = &monitors.back();
            mode = nullptr;
            header = trim(line.substr(markerPosition + 8));
            connectorEnd = header.find_first_of(" (");
            monitor->connector = header.substr(0, connectorEnd);
            if (connectorEnd != std::string::npos)
            {
                markerPosition = header.find('(', connectorEnd);
                if (markerPosition != std::string::npos &&
                    header.back() == ')')
                {
                    monitor->displayName = header.substr(
                        markerPosition + 1,
                        header.size() - markerPosition - 2);
                }
            }
        }
        else if (monitor != nullptr &&
                 line.find("Product:") != std::string_view::npos)
        {
            monitor->productName = valueAfter(line, "Product:");
        }
        else if (monitor != nullptr &&
                 line.find("display-name") != std::string_view::npos)
        {
            monitor->displayName = valueAfter(line, "display-name");
        }
        else if (monitor != nullptr &&
                 line.find("min-refresh-rate") != std::string_view::npos)
        {
            parsePositiveFloat(valueAfter(line, "min-refresh-rate"),
                               &monitor->minRefreshRateHz);
        }
        else if (monitor != nullptr &&
                 line.find("Refresh rate:") != std::string_view::npos)
        {
            refreshRate = 0.0f;
            if (parsePositiveFloat(valueAfter(line, "Refresh rate:"),
                                   &refreshRate))
            {
                monitor->modes.emplace_back();
                mode = &monitor->modes.back();
                mode->refreshRateHz = refreshRate;
            }
        }
        else if (mode != nullptr &&
                 line.find("is-current") != std::string_view::npos)
        {
            mode->current = valueAfter(line, "is-current") == "yes";
        }
        else if (mode != nullptr &&
                 line.find("refresh-rate-mode") != std::string_view::npos)
        {
            mode->variable =
                valueAfter(line, "refresh-rate-mode") == "variable";
        }
        lineStart = lineEnd + 1;
    }

    if (monitors.empty())
    {
        return {};
    }
    if (monitors.size() == 1)
    {
        return stateFromMonitor(monitors[0]);
    }
    while (index < monitors.size())
    {
        score = monitorMatchScore(monitors[index], monitorName);
        if (score > bestScore)
        {
            bestScore = score;
            bestMonitor = &monitors[index];
        }
        ++index;
    }
    if (bestMonitor == nullptr || bestScore <= 0)
    {
        return {};
    }
    return stateFromMonitor(*bestMonitor);
}

PlatformDisplayState probePlatformDisplay(std::string_view monitorName,
                                          void *nativeWindow)
{
#if defined(__linux__)
    (void)nativeWindow;
    return probeGdctl(monitorName);
#elif defined(_WIN32)
    PlatformDisplayState result;
    HWND window = static_cast<HWND>(nativeWindow);
    HMONITOR monitor = nullptr;
    MONITORINFOEXW monitorInfo = {};
    DEVMODEW mode = {};

    (void)monitorName;
    if (window == nullptr)
    {
        return result;
    }
    monitor = MonitorFromWindow(window, MONITOR_DEFAULTTONEAREST);
    monitorInfo.cbSize = sizeof(monitorInfo);
    mode.dmSize = sizeof(mode);
    if (monitor == nullptr ||
        GetMonitorInfoW(monitor, &monitorInfo) == FALSE ||
        EnumDisplaySettingsW(monitorInfo.szDevice, ENUM_CURRENT_SETTINGS,
                             &mode) == FALSE)
    {
        return result;
    }
    if (mode.dmDisplayFrequency > 1)
    {
        result.currentRefreshRateHz =
            static_cast<float>(mode.dmDisplayFrequency);
    }
    return result;
#else
    (void)monitorName;
    (void)nativeWindow;
    return {};
#endif
}

} // namespace display_output
} // namespace oka
