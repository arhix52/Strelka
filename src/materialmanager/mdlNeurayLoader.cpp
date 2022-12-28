//    Copyright (C) 2021 Pablo Delgado Krämer
//
//        This program is free software: you can redistribute it and/or modify
//    it under the terms of the GNU General Public License as published by
//    the Free Software Foundation, either version 3 of the License, or
//                                         (at your option) any later version.
//
//                                         This program is distributed in the hope that it will be useful,
//    but WITHOUT ANY WARRANTY; without even the implied warranty of
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
//    GNU General Public License for more details.
//
//    You should have received a copy of the GNU General Public License
//    along with this program. If not, see <https://www.gnu.org/licenses/>.

#include "mdlNeurayLoader.h"

#include <mi/mdl_sdk.h>

#ifdef MI_PLATFORM_WINDOWS
#    include <mi/base/miwindows.h>
#else
#    include <dlfcn.h>
#endif

#include "iostream"

#include <string>

namespace oka
{
MdlNeurayLoader::MdlNeurayLoader()
    : mDsoHandle(nullptr), mNeuray(nullptr)
{
}

MdlNeurayLoader::~MdlNeurayLoader()
{
    if (mNeuray)
    {
        mNeuray->shutdown();
        mNeuray.reset();
    }
    if (mDsoHandle)
    {
        unloadDso();
    }
}

bool MdlNeurayLoader::init(const char* resourcePath, const char* imagePluginPath)
{
    if (!loadDso(resourcePath))
    {
        return false;
    }
    if (!loadNeuray())
    {
        return false;
    }
    if (!loadPlugin(imagePluginPath))
    {
        return false;
    }

    return mNeuray->start() == 0;
}

mi::base::Handle<mi::neuraylib::INeuray> MdlNeurayLoader::getNeuray() const
{
    return mNeuray;
}

bool MdlNeurayLoader::loadPlugin(const char* imagePluginPath)
{
    // init plugin for texture support
    mi::base::Handle<mi::neuraylib::INeuray> neuray(getNeuray());
    mi::base::Handle<mi::neuraylib::IPlugin_configuration> configPl(neuray->get_api_component<mi::neuraylib::IPlugin_configuration>());
    if (configPl->load_plugin_library(imagePluginPath)) // This function can only be called before the MDL SDK has been started.
    {
        std::cout << "Plugin file path not found, translation not possible" << std::endl;
        return false;
    }

    return true;
}

bool MdlNeurayLoader::loadDso(const char* resourcePath)
{
    std::string dsoFilename = std::string(resourcePath) + std::string("/libmdl_sdk" MI_BASE_DLL_FILE_EXT);

#ifdef MI_PLATFORM_WINDOWS
    HMODULE handle = LoadLibraryA(dsoFilename.c_str());
    if (!handle)
    {
        LPTSTR buffer = NULL;
        LPCTSTR message = TEXT("unknown failure");
        DWORD error_code = GetLastError();
        if (FormatMessage(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM |
                              FORMAT_MESSAGE_IGNORE_INSERTS,
                          0, error_code,
                          MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), (LPTSTR)&buffer, 0, 0))
        {
            message = buffer;
        }
        fprintf(stderr, "Failed to load library (%u): %s", error_code, message);
        if (buffer)
        {
            LocalFree(buffer);
        }
        return false;
    }
#else
    void* handle = dlopen(dsoFilename.c_str(), RTLD_LAZY);
    if (!handle)
    {
        fprintf(stderr, "%s\n", dlerror());
        return false;
    }
#endif

    mDsoHandle = handle;
    return true;
}

bool MdlNeurayLoader::loadNeuray()
{
#ifdef MI_PLATFORM_WINDOWS
    void* symbol = GetProcAddress(reinterpret_cast<HMODULE>(mDsoHandle), "mi_factory");
    if (!symbol)
    {
        LPTSTR buffer = NULL;
        LPCTSTR message = TEXT("unknown failure");
        DWORD error_code = GetLastError();
        if (FormatMessage(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM |
                              FORMAT_MESSAGE_IGNORE_INSERTS,
                          0, error_code,
                          MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), (LPTSTR)&buffer, 0, 0))
        {
            message = buffer;
        }
        fprintf(stderr, "GetProcAddress error (%u): %s", error_code, message);
        if (buffer)
        {
            LocalFree(buffer);
        }
        return false;
    }
#else
    void* symbol = dlsym(mDsoHandle, "mi_factory");
    if (!symbol)
    {
        fprintf(stderr, "%s\n", dlerror());
        return false;
    }
#endif

    mNeuray = mi::base::Handle<mi::neuraylib::INeuray>(mi::neuraylib::mi_factory<mi::neuraylib::INeuray>(symbol));
    if (mNeuray.is_valid_interface())
    {
        return true;
    }

    mi::base::Handle<mi::neuraylib::IVersion> version(mi::neuraylib::mi_factory<mi::neuraylib::IVersion>(symbol));
    if (!version)
    {
        fprintf(stderr, "Error: Incompatible library.\n");
    }
    else
    {
        fprintf(stderr, "Error: Library version %s does not match header version %s.\n", version->get_product_version(), MI_NEURAYLIB_PRODUCT_VERSION_STRING);
    }

    return false;
}

void MdlNeurayLoader::unloadDso()
{
#ifdef MI_PLATFORM_WINDOWS
    if (FreeLibrary(reinterpret_cast<HMODULE>(mDsoHandle)))
    {
        return;
    }
    LPTSTR buffer = 0;
    LPCTSTR message = TEXT("unknown failure");
    DWORD error_code = GetLastError();
    if (FormatMessage(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM |
                          FORMAT_MESSAGE_IGNORE_INSERTS,
                      0, error_code,
                      MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), (LPTSTR)&buffer, 0, 0))
    {
        message = buffer;
    }
    fprintf(stderr, "Failed to unload library (%u): %s", error_code, message);
    if (buffer)
    {
        LocalFree(buffer);
    }
#else
    if (dlclose(mDsoHandle) != 0)
    {
        printf("%s\n", dlerror());
    }
#endif
}
} // namespace oka
