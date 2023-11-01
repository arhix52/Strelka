#pragma once

#include <glad/glad.h>

#include "Display.h"

namespace oka
{

class glfwdisplay : public Display
{
private:
    /* data */
    GLuint m_render_tex = 0u;
    GLuint m_program = 0u;
    GLint m_render_tex_uniform_loc = -1;
    GLuint m_quad_vertex_buffer = 0;
    GLuint m_dislpayPbo = 0;

    static const std::string s_vert_source;
    static const std::string s_frag_source;

public:
    glfwdisplay(/* args */);
    virtual ~glfwdisplay();

public:
    virtual void init(int width, int height, oka::SharedContext* ctx) override;
    void destroy();

    void onBeginFrame();
    void onEndFrame();

    void drawFrame(ImageBuffer& result);
    void drawUI();

    void display(const int32_t screen_res_x,
                 const int32_t screen_res_y,
                 const int32_t framebuf_res_x,
                 const int32_t framebuf_res_y,
                 const uint32_t pbo) const;
};
} // namespace oka
