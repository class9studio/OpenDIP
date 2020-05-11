# Set the minimum version of CMake that can be used
cmake_minimum_required(VERSION 3.8)

# Set the project name
project (OpenDIP LANGUAGES CXX CUDA)

# Use c++11, to build catch2 
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set Options
option(WITH_OPENCV "With OpenCV Support" ON)
option(WITH_MATPLOTLIB_CPP "With Matplotlib-cpp Support" ON)
option(WITH_CUDA_CPP "With Cuda-cpp Support" ON)

# add the CMakeFile that defines catch2、eigen3
add_subdirectory(3rd_party/catch2)

# Create a library to all cpp files to compile
set(COMMON_SOURCE
    src/common/common.cpp 
	src/common/image.cpp
)

set(ALGORITHM_SOURCE
    src/algorithm/img_basic.cpp
    src/algorithm/gray_transform.cpp
    src/algorithm/spatial_filtering.cpp 
    src/algorithm/img_segmentation.cpp 
    src/algorithm/img_morphology.cpp
    src/algorithm/frequency_filtering.cpp
    src/algorithm/img_features.cpp

    #src/algorithm/features_extraction/harris_opencv.cpp   #OpenCV only
    src/algorithm/features_extraction/hog_svm_opencv.cpp
)

set(CUDA_CPP_SOURCE
    src/cuda/cudaCommon.cu
    src/cuda/cudaConv2d.cu
    src/cuda/cudaImgResize.cu
    src/cuda/cudaRedutionSum.cu
    src/cuda/cudaThreadsOrg.cu
    src/cuda/cudaUniqueGid.cu
    src/cuda/cudaGrid2D.cu
    src/cuda/cudaMemTrs.cu
    src/cuda/cudaDeviceQuery.cu
    src/cuda/cudaSumArray.cu
    src/cuda/cudaWarpIndex.cu
    src/cuda/cudaWarpDivergence.cu
    src/cuda/cudaReductionNeiPair.cu
    src/cuda/cudaReductionInterleavedPair.cu
    src/cuda/cudaReductionWarpDivImp.cu
    # src/cuda/cudaReductionWithDynamicPara.cu
    src/cuda/cudaReductionComparison.cu
    src/cuda/cudaMatrixTranspose.cu
    )

add_library(OpenDIP STATIC
    ${COMMON_SOURCE}
    ${ALGORITHM_SOURCE}
    ${CUDA_CPP_SOURCE}
)

if(WIN32)	
    message(STATUS "Now is windows")
elseif(APPLE)	
    message(STATUS "Now is Apple systens.")
elseif(UNIX)	
    message(STATUS "Now is UNIX-like OS's.")
    # include third package library, then check
    find_package (Eigen3  REQUIRED NO_MODULE)
    if(Eigen3_FOUND)
        message ("Eigen3 found")
        set(EIGEN3_INCLUDE_DIR ${EIGEN_INCLUDE_DIRS})
    else()
        message (FATAL_ERROR "Cannot find Eigen3, then install")
        add_subdirectory(3rd_party/eigen3)
        set(EIGEN3_INCLUDE_DIR ${EIGEN_INCLUDE_DIRS})
    endif()

    if(WITH_OPENCV)
        find_package(OpenCV REQUIRED)
        if(OpenCV_FOUND)
            message("OpenCV found")
            target_link_libraries(OpenDIP
            ${OpenCV_LIBS}) 
            target_include_directories(OpenDIP PUBLIC 
                ${OpenCV_INCLUDE_DIRS}) 
        else()
            message(FATAL_ERROR "cannot find OpenCV")
        endif()
    endif()  

    if(WITH_MATPLOTLIB_CPP)
        find_package(PythonLibs 2.7)
        if(PythonLibs_FOUND)
            message("PythonLibs found")
            target_link_libraries(OpenDIP 
                ${PYTHON_LIBRARIES})    
            target_include_directories(OpenDIP PUBLIC 
                ${PYTHON_INCLUDE_DIRS})
            add_definitions(-DMATPLOTLIBCPP_PYTHON_HEADER=Python.h)
        else()
            message(FATAL_ERROR "cannot find PythonLibs")
        endif()
    endif()    
endif()

#############################################
# Unit tests

# enable CTest testing
enable_testing()

# Create a sources variable with a link to all cpp files to compile
set(SOURCES
    unit_tests/unit_tests.cpp
)
# Set the directories that should be included in the build command for this target
# when running g++ these will be included as -I/directory/path/

# Add a testing executable
add_executable(unit_tests ${SOURCES})


# link library
target_link_libraries(unit_tests
    OpenDIP
    Catch2::Test
)

add_test(test_all unit_tests)

# Set the directories that should be included in the build command for this target
# when running g++ these will be included as -I/directory/path/
target_include_directories(OpenDIP
    PUBLIC 
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/3rd_party/stb
        ${PROJECT_SOURCE_DIR}/3rd_party/catch2
        ${PROJECT_SOURCE_DIR}/3rd_party/eigen3
		${EIGEN_INCLUDE_DIRS}
)


