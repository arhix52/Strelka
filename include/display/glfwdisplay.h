#include <render/common.h>
#include <render/buffer.h>

#include <glad/gl.h>
#include <GLFW/glfw3.h>

namespace oka
{
class InputHandler
{
public:
    virtual void keyCallback(int key, [[maybe_unused]] int scancode, int action, [[maybe_unused]] int mods) = 0;
    virtual void mouseButtonCallback(int button, int action, [[maybe_unused]] int mods) = 0;
    virtual void handleMouseMoveCallback([[maybe_unused]] double xpos, [[maybe_unused]] double ypos) = 0;
};

class ResizeHandler
{
public:
    virtual void framebufferResize(int newWidth, int newHeight) = 0;
};

class glfwdisplay
{
private:
    /* data */
    GLuint m_render_tex = 0u;
    GLuint m_program = 0u;
    GLint m_render_tex_uniform_loc = -1;
    GLuint m_quad_vertex_buffer = 0;
    GLuint m_dislpayPbo = 0;

    // BufferImageFormat m_image_format;

    static const std::string s_vert_source;
    static const std::string s_frag_source;

    int mWindowWidth = 800;
    int mWindowHeight = 600;

    GLFWwindow* mWindow;

    InputHandler* mInputHandler = nullptr;
    ResizeHandler* mResizeHandler = nullptr;

    oka::SharedContext* mCtx = nullptr;

    static void framebufferResizeCallback(GLFWwindow* window, int width, int height);
    static void keyCallback(
        GLFWwindow* window, int key, [[maybe_unused]] int scancode, int action, [[maybe_unused]] int mods);
    static void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods);
    static void handleMouseMoveCallback(GLFWwindow* window, double xpos, double ypos);
    static void scrollCallback(GLFWwindow* window, double xoffset, double yoffset);

public:
    glfwdisplay(/* args */);
    virtual ~glfwdisplay();

public:
    void init(int width, int height, oka::SharedContext* ctx);
    void destroy();

    void setWindowTitle(const char* title);

    void setInputHandler(InputHandler* handler)
    {
        mInputHandler = handler;
    }
    InputHandler* getInputHandler()
    {
        return mInputHandler;
    }

    void setResizeHandler(ResizeHandler* handler)
    {
        mResizeHandler = handler;
    }
    ResizeHandler* getResizeHandler()
    {
        return mResizeHandler;
    }

    // Ui& Ui()
    // {
    //     return mUi;
    // }

    bool windowShouldClose();
    void pollEvents();

    void onBeginFrame();
    void onEndFrame();

    void drawFrame(ImageBuffer& result);
    void drawUI();

    bool framebufferResized = false;


    void display(const int32_t screen_res_x,
                 const int32_t screen_res_y,
                 const int32_t framebuf_res_x,
                 const int32_t framebuf_res_y,
                 const uint32_t pbo) const;
};
} // namespace oka
