#include <log/log.h>
#include <logmanager.h>
#include <cxxopts.hpp>

#include <filesystem>

#include "OpenXRHelpers.h"
#include "OpenXRProgram.h"

int main(int argc, const char* argv[])
{
    const oka::Logmanager loggerManager;
    std::shared_ptr<oka::IOpenXrProgram> openXrProgram = oka::CreateOpenXrProgram();

    openXrProgram->CreateInstance();
    openXrProgram->InitializeSystem();

    openXrProgram->InitializeDevice();
    
    return 0;
}
