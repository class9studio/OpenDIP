# Set the minimum version of CMake that can be used
cmake_minimum_required(VERSION 3.1)

# Set the project name
project (OpenDIP)

# Create a sources variable with a link to all cpp files to compile
set(SOURCES
    src/common.cpp
    src/image.cpp
    src/unittest.cpp
    src/main.cpp
)

# Add an executable with the above sources
add_executable(OpenDIP ${SOURCES})

# Set the directories that should be included in the build command for this target
# when running g++ these will be included as -I/directory/path/
target_include_directories(OpenDIP
    PRIVATE 
        ${PROJECT_SOURCE_DIR}/include
)
