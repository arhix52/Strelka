# Strelka

## Project Dependencies

- Vulkan  - https://vulkan.lunarg.com/   *implicit*
- glfw    - https://www.glfw.org/     *dll*
- slang      - https://github.com/shader-slang/slang *dll*
- cxxopts   - https://github.com/jarro2783/cxxopts  *header*
- json - https://github.com/nlohmann/json *header*
- imgui   - https://github.com/ocornut/imgui *header+source*
- glm      - https://github.com/g-truc/glm *submodule*
- stb       - https://github.com/nothings/stb *submodule*
- tinygltf    - https://github.com/syoyo/tinygltf *submodule*
- tol - https://github.com/tinyobjloader/tinyobjloader *submodule*
- doctest      - https://github.com/onqtam/doctest *submodule*

## OSX Guide

#### Installation
Follow setup guide https://vulkan-tutorial.com/Development_environment

Clone the project.
   
    git clone https://github.com/ikryukov/NeVK --recursive

#### Launch
Use vscode with preset env variable
1. export VULKAN_SDK=~/vulkansdk/macOS
2. launch code 
    
## Synopsis 

    Strelka -s <USD Scene path> [OPTION...] positional parameters

    -s, --scene arg       scene path (default: "")
    -i, --iteration arg  Iteration to capture (default: -1)
    -h, --help            Print usage

## Example

    ./Strelka -s misc/coffeemaker.usdc -i 100

## USD
    Vulkan:
        cd <VULKAN_SDK>
        source ./setup-env.sh
    USD env:
        export USD_DIR=/Users/ilya/work/usd_build/
        export PATH=/Users/ilya/work/usd_build/bin:$PATH
        export PYTHONPATH=/Users/ilya/work/usd_build/lib/python:$PYTHONPATH

    Cmake:
        cmake -DCMAKE_INSTALL_PREFIX=/Users/ilya/work/usd_build/plugin/usd/ ..
    Install plugin:
        cmake --install . --component HdStrelka

## License
* USD plugin design and material translation code based on Pablo Gatling code:
https://github.com/pablode/gatling