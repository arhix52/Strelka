#pragma once

#include <glad/glad.h>

#include "Display.h"

namespace oka
{

class GlfwDisplay : public Display
{
private:
    GLuint m_render_tex = 0u;
    GLuint m_program = 0u;
    GLint m_render_tex_uniform_loc = -1;
    GLuint m_quad_vertex_buffer = 0;
    GLuint m_dislpayPbo = 0;

    static const std::string s_vert_source;
    static const std::string s_frag_source;

public:
    GlfwDisplay();
    virtual ~GlfwDisplay();

    virtual void init(int width, int height, SettingsManager* settings) override;
    void destroy();

    void onBeginFrame();
    void onEndFrame();

    void drawFrame(ImageBuffer& result);
    void drawUI();

    void* getDisplayNativeTexure() override;

    void display(const int32_t screen_res_x,
                 const int32_t screen_res_y,
                 const int32_t framebuf_res_x,
                 const int32_t framebuf_res_y,
                 const uint32_t pbo) const;
};
} // namespace oka
