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

#include <mi/base/handle.h>
#include <mi/neuraylib/ineuray.h>

namespace oka
{
class MdlNeurayLoader
{
public:
    MdlNeurayLoader();
    ~MdlNeurayLoader();

public:
    bool init(const char* resourcePath, const char* imagePluginPath);

    mi::base::Handle<mi::neuraylib::INeuray> getNeuray() const;

private:
    bool loadDso(const char* resourcePath);
    bool loadNeuray();
    bool loadPlugin(const char* imagePluginPath);
    void unloadDso();

private:
    void* mDsoHandle;
    mi::base::Handle<mi::neuraylib::INeuray> mNeuray;
};
}
