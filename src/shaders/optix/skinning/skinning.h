#pragma once
#include <sutil/Matrix.h>
#include <stdint.h>

extern "C" void cuApplySkinning(int threads_per_block,
                                const int vbOffset,
                                const int sbOffset,
                                void* vertexPtr,
                                const void* vertexSkinDataPtr,
                                const sutil::Matrix4x4* d_jointMats,
                                int jointMatOffset,
                                int jointCount,
                                const uint32_t vertexCount);
