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

#pragma once

#include "mdlLogger.h"
#include "mdlNeurayLoader.h"

#include <mi/base/handle.h>
#include <mi/neuraylib/idatabase.h>
#include <mi/neuraylib/itransaction.h>
#include <mi/neuraylib/imdl_backend_api.h>
#include <mi/neuraylib/imdl_impexp_api.h>
#include <mi/neuraylib/imdl_factory.h>

#include <mi/mdl_sdk.h>

#include <memory>
#include <vector>

namespace oka
{
class MdlRuntime
{
public:
    MdlRuntime();
    ~MdlRuntime();

public:
    bool init(const char* paths[], uint32_t numPaths, const char* neurayPath, const char* imagePluginPath);

    mi::base::Handle<MdlLogger> getLogger();
    mi::base::Handle<mi::neuraylib::IDatabase> getDatabase();
    mi::base::Handle<mi::neuraylib::ITransaction> getTransaction();
    mi::base::Handle<mi::neuraylib::IMdl_factory> getFactory();
    mi::base::Handle<mi::neuraylib::IMdl_impexp_api> getImpExpApi();
    mi::base::Handle<mi::neuraylib::IMdl_backend_api> getBackendApi();
    mi::base::Handle<mi::neuraylib::INeuray> getNeuray();
    
    mi::base::Handle<mi::neuraylib::IMdl_configuration> getConfig();

    std::unique_ptr<MdlNeurayLoader> mLoader;

private:
    mi::base::Handle<MdlLogger> mLogger;
    mi::base::Handle<mi::neuraylib::IDatabase> mDatabase;
    mi::base::Handle<mi::neuraylib::ITransaction> mTransaction;
    mi::base::Handle<mi::neuraylib::IMdl_factory> mFactory;
    mi::base::Handle<mi::neuraylib::IMdl_backend_api> mBackendApi;
    mi::base::Handle<mi::neuraylib::IMdl_impexp_api> mImpExpApi;

    mi::base::Handle<mi::neuraylib::IMdl_configuration> mConfig;
};
} // namespace oka
