# Strelka

## Project Dependencies

Strelka uses conan https://conan.io/
install conan: `pip install conan` 

detect conan profile: `conan profile detect --force`

1. `conan install . --output-folder=build --build=missing --settings=build_type=Debug`
2. `cd build`
3. `cmake .. -G "Visual Studio 17 2022" -DCMAKE_TOOLCHAIN_FILE=conan_toolchain.cmake --settings=build_type=Debug`

### Libs:
- glfw    - https://www.glfw.org/     *dll*
- cxxopts   - https://github.com/jarro2783/cxxopts  *header*
- imgui   - https://github.com/ocornut/imgui *header+source*
- glm      - https://github.com/g-truc/glm *submodule*
- stb       - https://github.com/nothings/stb *submodule*
- doctest      - https://github.com/onqtam/doctest *submodule*

#### Installation
Follow setup guide https://vulkan-tutorial.com/Development_environment

Clone the project.
   

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
        export USD_DIR=/Users/<user>/work/usd_build/
        export PATH=/Users/<user>/work/usd_build/bin:$PATH
        export PYTHONPATH=/Users/<user>/work/usd_build/lib/python:$PYTHONPATH

    Cmake:
        cmake -DCMAKE_INSTALL_PREFIX=/Users/<user>/work/usd_build/plugin/usd/ ..
    Install plugin:
        cmake --install . --component HdStrelka

## License
* USD plugin design and material translation code based on Pablo Gatling code:
https://github.com/pablode/gatling