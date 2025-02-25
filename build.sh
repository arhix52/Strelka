#!/bin/bash

if [ "$#" -ne 1 ] && [ "$#" -ne 2 ]; then
    echo "Usage: $0 <build_type> [clean]"
    exit 1
fi

build_type="$1"
clean_option="$2"

# Function to convert the input parameter to start with a capital letter
ucfirst() {
    echo "$1" | awk '{print toupper(substr($0,1,1)) tolower(substr($0,2))}'
}

# Convert the build_type to start with a capital letter
build_type=$(ucfirst "$build_type")

# init submodules
git submodule update --init --recursive

# Step 1: Install Conan dependencies
conan install . -c tools.cmake.cmaketoolchain:generator=Ninja -c tools.system.package_manager:mode=install -c tools.system.package_manager:sudo=True --build=missing --settings=build_type="$build_type"

# Step 2: Navigate to the build directory
cd build/"$build_type"

# Check if the "clean" option is specified
if [ "$clean_option" == "clean" ]; then
    # Clean the build directory
    cmake --build . --target clean
fi

# Step 3: Source the Conan environment variables
source ./generators/conanbuild.sh

# Step 4: Run CMake with the appropriate toolchain file
cmake ../.. -G Ninja -DCMAKE_TOOLCHAIN_FILE=generators/conan_toolchain.cmake -DCMAKE_BUILD_TYPE="$build_type"

# Step 5: Build the project and capture the time
start_time=$(date +%s)
cmake --build .
end_time=$(date +%s)

# Calculate the elapsed time
elapsed_time=$((end_time - start_time))

# Output the build time
echo "Build completed in $elapsed_time seconds."
