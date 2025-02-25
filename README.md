# Strelka
Path tracing render based on NVIDIA OptiX + NVIDIA MDL and Apple Metal
## OpenUSD Hydra render delegate
![Kitchen Set from OpenUSD](images/Kitchen_2048i_4d_2048spp_0.png)
## Basis curves support
![Hairs](images/hairmat_2_light_10000i_6d_10000spp_0.png)
![Einar](images/einar_1024i_3d_1024spp_0.png)

## Project Dependencies

OpenUSD https://github.com/PixarAnimationStudios/OpenUSD
See BuildOpenUSD.md in repo

* Set evn var: `USD_DIR=c:\work\USD_build`

OptiX 
* Set evn var: `OPTIX_DIR=C:\work\OptiX SDK 8.0.0`

Download MDL sdk (for example: mdl-sdk-367100.2992): https://developer.nvidia.com/nvidia-mdl-sdk-get-started

* unzip content to /external/mdl-sdk/

LLVM 12.0.1 (https://github.com/llvm/llvm-project/releases/tag/llvmorg-12.0.1) for MDL ptx code generator

* for win: https://github.com/llvm/llvm-project/releases/download/llvmorg-12.0.1/LLVM-12.0.1-win64.exe
* for linux: https://github.com/llvm/llvm-project/releases/download/llvmorg-12.0.1/clang+llvm-12.0.1-x86_64-linux-gnu-ubuntu-16.04.tar.xz
* install it to `c:\work` for example
* add to PATH: `c:\work\LLVM\bin`
* extract 2 header files files from external/clang12_patched to `C:\work\LLVM\lib\clang\12.0.1\include`

Strelka uses conan https://conan.io/

* install conan: `pip install conan` 
* install ninja [https://ninja-build.org/] build system: `sudo apt install ninja-build`

* initialize submodules `git submodule update --init --recursive`

detect conan profile: `conan profile detect --force`

1. `conan install . --build=missing --settings=build_type=Debug`
2. `cd build`
3. `cmake .. -G "Visual Studio 17 2022" -DCMAKE_TOOLCHAIN_FILE=generators\conan_toolchain.cmake`
4. `cmake --build . --config Debug`

On Mac/Linux:
1. `conan install . -c tools.cmake.cmaketoolchain:generator=Ninja -c tools.system.package_manager:mode=install -c tools.system.package_manager:sudo=True --build=missing --settings=build_type=Debug`
2. `cd build/Debug`
3. `source ./generators/conanbuild.sh`
4. `cmake ../.. -DCMAKE_TOOLCHAIN_FILE=generators/conan_toolchain.cmake -DCMAKE_BUILD_TYPE=Debug`
5. `cmake --build .`

#### Installation

#### Launch
    
## Synopsis 

    Strelka -s <USD Scene path> [OPTION...] positional parameters

    -s, --scene arg       scene path (default: "")
    -i, --iteration arg  Iteration to capture (default: -1)
    -h, --help            Print usage


To set log level use

    `export SPDLOG_LEVEL=debug`
    
The available log levels are: trace, debug, info, warn, and err.

## Example

    ./Strelka -s misc/coffeemaker.usdc -i 100

## USD
    USD env:
        export USD_DIR=/Users/<user>/work/usd_build/
        export PATH=/Users/<user>/work/usd_build/bin:$PATH
        export PYTHONPATH=/Users/<user>/work/usd_build/lib/python:$PYTHONPATH

    Install plugin:
        cmake --install . --component HdStrelka

## License
* USD plugin design and material translation code based on Pablo Gatling code:
https://github.com/pablode/gatling