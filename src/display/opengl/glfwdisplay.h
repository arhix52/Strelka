#pragma once

#include <glad/glad.h>

#include "Display.h"

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

namespace oka
{

class GlfwDisplay : public Display
{
private:
    // OpenGL texture for CUDA rendering
    cudaGraphicsResource* cudaResource;
    GLuint m_render_tex = 0u;

public:
    GlfwDisplay();
    ~GlfwDisplay() override;

    virtual void init(int width, int height, SettingsManager* settings) override;
    void destroy() override;

    void onBeginFrame() override;
    void onEndFrame() override;

    void drawFrame(ImageBuffer& result) override;
    void drawUI() override;

    void* getDisplayNativeTexure() override;
    float getMaxEDR() override;
};
} // namespace oka
