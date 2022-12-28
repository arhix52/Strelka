//
// Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//

#pragma once

#include <cuda_runtime.h>


struct MaterialData
{
    enum Type
    {
        PBR = 0
    };

    MaterialData()
    {
        type                       = MaterialData::PBR;
        pbr.base_color             = { 1.0f, 1.0f, 1.0f, 1.0f };
        pbr.metallic               = 1.0f;
        pbr.roughness              = 1.0f;
        pbr.base_color_tex         = { 0, 0 };
        pbr.metallic_roughness_tex = { 0, 0 };
    }

    enum AlphaMode
    {
        ALPHA_MODE_OPAQUE = 0,
        ALPHA_MODE_MASK   = 1,
        ALPHA_MODE_BLEND  = 2
    };

    struct Texture
    {
        __device__ __forceinline__ operator bool() const
        {
            return tex != 0;
        }

        int                  texcoord;
        cudaTextureObject_t  tex;

        float2               texcoord_offset;
        float2               texcoord_rotation; // sin,cos
        float2               texcoord_scale;
    };

    struct Pbr
    {
        float4               base_color;
        float                metallic;
        float                roughness;

        Texture              base_color_tex;
        Texture              metallic_roughness_tex;
    };

    Type                 type            = PBR;

    Texture              normal_tex      = { 0 , 0 };

    AlphaMode            alpha_mode      = ALPHA_MODE_OPAQUE;
    float                alpha_cutoff    = 0.f;

    float3               emissive_factor = { 0.f, 0.f, 0.f };
    Texture              emissive_tex    = { 0, 0 };

    bool                 doubleSided     = false;

    union
    {
        Pbr  pbr;
    };
};
