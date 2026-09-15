#pragma once

// Frozen 128x128 void-and-cluster rank mask shared byte-for-byte with CUDA.
constant const ushort kBlueNoiseRank[16384] = {
#include "bluenoise_rank.inc"
};
