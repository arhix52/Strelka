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

#include <string>

#include <mi/base/handle.h>
#include <mi/neuraylib/icompiled_material.h>
#include <mi/neuraylib/idatabase.h>
#include <mi/neuraylib/itransaction.h>
#include <mi/neuraylib/imaterial_instance.h>
#include <mi/neuraylib/imdl_impexp_api.h>
#include <mi/neuraylib/imdl_execution_context.h>
#include <mi/neuraylib/imdl_factory.h>

#include "mdlRuntime.h"

namespace oka
{
class MdlMaterialCompiler
{
public:
    MdlMaterialCompiler(MdlRuntime& runtime);

public:
    bool createModule(const std::string& identifier,
                      std::string& moduleName);

    bool createMaterialInstace(const char* moduleName, const char* identifier, 
        mi::base::Handle<mi::neuraylib::IFunction_call>& matInstance);

    bool compileMaterial(mi::base::Handle<mi::neuraylib::IFunction_call>& instance,
                         mi::base::Handle<mi::neuraylib::ICompiled_material>& compiledMaterial);

    mi::base::Handle<mi::neuraylib::IMdl_factory>& getFactory();
    mi::base::Handle<mi::neuraylib::ITransaction>& getTransaction();

private:
    mi::base::Handle<MdlLogger> mLogger;
    mi::base::Handle<mi::neuraylib::IDatabase> mDatabase;
    mi::base::Handle<mi::neuraylib::ITransaction> mTransaction;
    mi::base::Handle<mi::neuraylib::IMdl_factory> mFactory;
    mi::base::Handle<mi::neuraylib::IMdl_impexp_api> mImpExpApi;
};
}
