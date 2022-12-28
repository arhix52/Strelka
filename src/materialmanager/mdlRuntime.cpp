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

#include "mdlRuntime.h"

#include <mi/mdl_sdk.h>

#include <vector>

namespace oka
{
MdlRuntime::MdlRuntime()
{
}

MdlRuntime::~MdlRuntime()
{
    if (mTransaction)
    {
        mTransaction->commit();
    }
}

bool MdlRuntime::init(const char* paths[], uint32_t numPaths, const char* neurayPath, const char* imagePluginPath)
{
    mLoader = std::make_unique<MdlNeurayLoader>();
    if (!mLoader->init(neurayPath, imagePluginPath))
    {
        return false;
    }

    mi::base::Handle<mi::neuraylib::INeuray> neuray(mLoader->getNeuray());
    mi::base::Handle<mi::neuraylib::IMdl_configuration> config(neuray->get_api_component<mi::neuraylib::IMdl_configuration>());

    mLogger = mi::base::Handle<MdlLogger>(new MdlLogger());
    config->set_logger(mLogger.get());

    for (uint32_t i = 0; i < numPaths; i++)
    {
        if (config->add_mdl_path(paths[i]) != 0 || config->add_resource_path(paths[i]) != 0)
        {
            mLogger->message(mi::base::MESSAGE_SEVERITY_FATAL, "MDL file path not found, translation not possible");
            return false;
        }
    }

    mDatabase = mi::base::Handle<mi::neuraylib::IDatabase>(neuray->get_api_component<mi::neuraylib::IDatabase>());
    mi::base::Handle<mi::neuraylib::IScope> scope(mDatabase->get_global_scope());
    mTransaction = mi::base::Handle<mi::neuraylib::ITransaction>(scope->create_transaction());

    mFactory = mi::base::Handle<mi::neuraylib::IMdl_factory>(neuray->get_api_component<mi::neuraylib::IMdl_factory>());
    mImpExpApi = mi::base::Handle<mi::neuraylib::IMdl_impexp_api>(neuray->get_api_component<mi::neuraylib::IMdl_impexp_api>());
    mBackendApi = mi::base::Handle<mi::neuraylib::IMdl_backend_api>(neuray->get_api_component<mi::neuraylib::IMdl_backend_api>());
    return true;
}

mi::base::Handle<mi::neuraylib::INeuray> MdlRuntime::getNeuray()
{
    return mLoader->getNeuray();
}

mi::base::Handle<MdlLogger> MdlRuntime::getLogger()
{
    return mLogger;
}

mi::base::Handle<mi::neuraylib::IDatabase> MdlRuntime::getDatabase()
{
    return mDatabase;
}

mi::base::Handle<mi::neuraylib::ITransaction> MdlRuntime::getTransaction()
{
    return mTransaction;
}

mi::base::Handle<mi::neuraylib::IMdl_factory> MdlRuntime::getFactory()
{
    return mFactory;
}

mi::base::Handle<mi::neuraylib::IMdl_impexp_api> MdlRuntime::getImpExpApi()
{
    return mImpExpApi;
}

mi::base::Handle<mi::neuraylib::IMdl_backend_api> MdlRuntime::getBackendApi()
{
    return mBackendApi;
}
} // namespace oka
