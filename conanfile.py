import os

from conan import ConanFile
from conan.tools.cmake import cmake_layout
from conan.tools.files import copy


class StrelkaRecipe(ConanFile):
    settings = "os", "compiler", "build_type", "arch"
    generators = "CMakeToolchain", "CMakeDeps"

    def requirements(self):
        self.requires("glm/cci.20230113")
        self.requires("spdlog/[>=1.4.1]")
        self.requires("imgui/1.89.3")
        self.requires("glfw/3.3.8")
        self.requires("stb/cci.20230920")
        self.requires("glad/0.1.36")
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
