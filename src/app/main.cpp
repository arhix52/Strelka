#include <display/glfwdisplay.h>

#include <render/buffer.h>

#include <render/OptiXRender.h>

int main()
{
    oka::glfwdisplay display;

    uint32_t imageWidth = 800;
    uint32_t imageHeight = 600;

    display.init(imageWidth, imageHeight);

    oka::CUDAOutputBuffer<uchar4> outputBuffer(oka::CUDAOutputBufferType::CUDA_DEVICE, imageWidth, imageHeight);

    oka::OptiXRender render;

    render.init();
    
    render.launch(outputBuffer);

    oka::ImageBuffer outputImage;
    outputImage.data = outputBuffer.getHostPointer();
    outputImage.dataSize = outputBuffer.getHostDataSize();
    outputImage.height = outputBuffer.height();
    outputImage.width = outputBuffer.width();
    outputImage.pixel_format = oka::BufferImageFormat::UNSIGNED_BYTE4;

    while (!display.windowShouldClose())
    {
        display.pollEvents();

        display.onBeginFrame();

        display.drawFrame(outputImage); // blit rendered image to swapchain
        display.drawUI(); // render ui to swapchain image in window resolution
        display.onEndFrame();

        display.setWindowTitle("Strelka OptiX");
    }
    display.destroy();
    return 0;
}