import os

from conan import ConanFile
from conan.tools.cmake import cmake_layout
from conan.tools.files import copy


class StrelkaRecipe(ConanFile):
    settings = "os", "compiler", "build_type", "arch"
    generators = "CMakeToolchain", "CMakeDeps"

    def requirements(self):
        # Foundation
        self.requires("glm/cci.20230113")
        self.requires("spdlog/1.14.1")

        # Scene loading
        self.requires("tinygltf/2.8.19")
        self.requires("nlohmann_json/3.11.3")
        self.requires("stb/cci.20230920")
        self.requires("tinyexr/1.0.7")

        # Editor (conditional via options)
        # 1.92.9b-docking is the first release carrying imgui_impl_metal4, which is
        # what lets the UI pass move off Metal 3. conan-center has not published it
        # yet, so the recipe is exported locally -- see docs/imgui-metal4.md.
        self.requires("imgui/1.92.9b-docking", override=True)
        self.requires("glfw/3.4")
        # ImGuizmo's conan-center package is from 2023 and calls ImGui APIs that
        # 1.92 removed (BeginChildFrame, the old AddPolyline signature). Upstream
        # has kept up; this is a local export of its head. Bumping ImGui for the
        # Metal 4 backend forces this bump with it.
        self.requires("imguizmo/cci.20260729")
        if self.settings.os != "Macos":
            self.requires("glad/0.1.36")
        self.requires("cxxopts/3.1.1")

        # Testing
        self.requires("doctest/2.4.11")

    def generate(self):
        copy(self, "*glfw*", os.path.join(self.dependencies["imgui"].package_folder,
             "res", "bindings"), os.path.join(self.source_folder, "external", "imgui"))
        copy(self, "*opengl3*", os.path.join(self.dependencies["imgui"].package_folder,
             "res", "bindings"), os.path.join(self.source_folder, "external", "imgui"))
        copy(self, "*metal*", os.path.join(self.dependencies["imgui"].package_folder,
             "res", "bindings"), os.path.join(self.source_folder, "external", "imgui"))

    def layout(self):
        cmake_layout(self)
