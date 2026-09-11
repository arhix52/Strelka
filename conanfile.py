import os

from conan import ConanFile
from conan.tools.cmake import cmake_layout
from conan.tools.files import copy


class StrelkaRecipe(ConanFile):
    settings = "os", "compiler", "build_type", "arch"
    generators = "CMakeToolchain", "CMakeDeps"
    default_options = {
        "glfw/*:with_wayland": True,
        "glfw/*:with_x11": True,
        # GLM 1.0.3's Conan recipe builds a compiled library by default. That
        # bakes radians-vs-degrees and clip-space at library compile time, so
        # GLM_FORCE_RADIANS / GLM_FORCE_DEPTH_ZERO_TO_ONE in glm_wrapper.hpp
        # would not apply. Vertex and curve uploads also assume packed float3
        # (12 bytes); stay header-only so those defines reach every include.
        "glm/*:header_only": True,
        # tinyexr's scanline writer compresses one block at a time unless this is
        # on, and then it fans the blocks out over std::thread. A 4K frame is
        # 71 MB of deflate, and it was 83% of a StrelkaCLI run whose render took
        # 0.3 s. The option's name -- and tinyexr's own default comment -- only
        # mention threaded loading; the save path reads the same macro.
        "tinyexr/*:with_thread": True,
    }

    def requirements(self):
        # Foundation
        self.requires("glm/1.0.3")
        self.requires("spdlog/1.17.0")
        self.requires("tomlplusplus/3.4.0")

        # Scene loading
        self.requires("tinygltf/2.9.7")
        self.requires("nlohmann_json/3.12.0")
        self.requires("stb/cci.20240531")
        self.requires("tinyexr/1.0.7")

        # Editor (conditional via options)
        # 1.92.9b-docking is the first release carrying imgui_impl_metal4, which is
        # what lets the UI pass move off Metal 3. conan-center has not published it
        # yet, so the recipe is exported locally -- see docs/imgui-metal4.md.
        self.requires("imgui/1.92.9b-docking", override=True)
        # Conan Center has not published 3.5.1 yet; build.sh exports the
        # official release through scripts/export_local_conan.sh.
        self.requires("glfw/3.5.1")
        # ImGuizmo's conan-center package is from 2023 and calls ImGui APIs that
        # 1.92 removed (BeginChildFrame, the old AddPolyline signature). Upstream
        # has kept up; this is a local export of its head. Bumping ImGui for the
        # Metal 4 backend forces this bump with it.
        self.requires("imguizmo/cci.20260729")
        if self.settings.os != "Macos":
            self.requires("vulkan-loader/1.3.268.0")
        self.requires("cxxopts/3.3.1")

        # Testing
        self.requires("doctest/2.5.2")

    def build_requirements(self):
        if self.settings.os != "Macos":
            self.tool_requires("shaderc/2025.3")

    def generate(self):
        copy(self, "*glfw*", os.path.join(self.dependencies["imgui"].package_folder,
             "res", "bindings"), os.path.join(self.source_folder, "external", "imgui"))
        if self.settings.os != "Macos":
            copy(self, "*vulkan*", os.path.join(self.dependencies["imgui"].package_folder,
                 "res", "bindings"), os.path.join(self.source_folder, "external", "imgui"))
        copy(self, "*metal*", os.path.join(self.dependencies["imgui"].package_folder,
             "res", "bindings"), os.path.join(self.source_folder, "external", "imgui"))

    def layout(self):
        cmake_layout(self)
