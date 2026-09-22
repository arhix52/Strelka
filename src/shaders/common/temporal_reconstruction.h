#ifndef STRELKA_TEMPORAL_RECONSTRUCTION_H
#define STRELKA_TEMPORAL_RECONSTRUCTION_H

// Scalar helpers shared by the camera, motion-vector writer, MetalFX setup,
// and host tests. Keeping the signs here prevents the three conventions from
// drifting independently.
inline float strelkaCameraSampleCoordinate(float cameraJitter)
{
    return 0.5f + cameraJitter;
}

inline float strelkaMetalFxJitterOffset(float cameraJitter)
{
    return -cameraJitter;
}

inline float strelkaScreenMotionAxis(float previousProjectedPixel, float currentPixelCenter, float cameraJitter)
{
    return previousProjectedPixel - (currentPixelCenter + cameraJitter);
}

#endif
