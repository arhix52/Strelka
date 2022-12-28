#pragma once
#define GLFW_INCLUDE_NONE
#ifdef _WIN32
#    define GLFW_EXPOSE_NATIVE_WIN32
#endif

//#define GLFW_INCLUDE_VULKAN
//#include <GLFW/glfw3.h>
#define GLM_FORCE_SILENT_WARNINGS
#include <glm/glm.hpp>
#include <glm/gtx/compatibility.hpp>
#include <vulkan/vulkan.h>

namespace oka
{
namespace debug
{

void setupDebug(VkInstance instance);

void beginLabel(VkCommandBuffer cmdBuffer, const char* labelName, const glm::float4& color);

void endLabel(VkCommandBuffer cmdBuffer);

} // namespace debug
} // namespace oka
