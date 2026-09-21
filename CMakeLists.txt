cmake_minimum_required(VERSION 3.16)

project(RayTracingBenchmark LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Threads REQUIRED)

option(RT_ENABLE_EMBREE "Build the optional Embree CPU backend" OFF)
option(RT_ENABLE_CUDA "Build the hand-written CUDA renderer backend" OFF)
option(RT_ENABLE_OPTIX "Build the optional NVIDIA OptiX backend" OFF)

add_executable(raytracer_bench
    component/main.cpp
    render/normal_cpu/cpu_backend.cpp
)

target_include_directories(raytracer_bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(raytracer_bench PRIVATE Threads::Threads)

if(RT_ENABLE_CUDA OR RT_ENABLE_OPTIX)
    if(CMAKE_VERSION VERSION_LESS 3.17)
        message(FATAL_ERROR "CUDA and OptiX backends require CMake 3.17 or newer")
    endif()
    if(RT_ENABLE_CUDA)
        if(CMAKE_VERSION VERSION_LESS 3.18)
            message(FATAL_ERROR "RT_ENABLE_CUDA requires CMake 3.18 or newer")
        endif()
        set(RT_CUDA_ARCHITECTURES "86" CACHE STRING
            "CUDA architectures used by the hand-written CUDA backend")
        set(CMAKE_CUDA_ARCHITECTURES "${RT_CUDA_ARCHITECTURES}")
    endif()
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
endif()

if(RT_ENABLE_CUDA)
    target_sources(raytracer_bench PRIVATE render/cuda/cuda_backend.cu)
    target_link_libraries(raytracer_bench PRIVATE CUDA::cudart)
    target_compile_definitions(raytracer_bench PRIVATE RT_ENABLE_CUDA=1)
    set_target_properties(raytracer_bench PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
        CUDA_ARCHITECTURES "${RT_CUDA_ARCHITECTURES}")
endif()

if(RT_ENABLE_EMBREE)
    find_path(EMBREE4_INCLUDE_DIR NAMES embree4/rtcore.h)
    find_library(EMBREE4_LIBRARY NAMES embree4)
    find_path(EMBREE3_INCLUDE_DIR NAMES embree3/rtcore.h)
    find_library(EMBREE3_LIBRARY NAMES embree3)
    if(EMBREE4_INCLUDE_DIR AND EMBREE4_LIBRARY)
        set(EMBREE_INCLUDE_DIR "${EMBREE4_INCLUDE_DIR}")
        set(EMBREE_LIBRARY "${EMBREE4_LIBRARY}")
        set(RT_EMBREE_VERSION_MAJOR 4)
    elseif(EMBREE3_INCLUDE_DIR AND EMBREE3_LIBRARY)
        set(EMBREE_INCLUDE_DIR "${EMBREE3_INCLUDE_DIR}")
        set(EMBREE_LIBRARY "${EMBREE3_LIBRARY}")
        set(RT_EMBREE_VERSION_MAJOR 3)
    else()
        message(FATAL_ERROR "RT_ENABLE_EMBREE=ON requires matching Embree 3 or 4 headers and library")
    endif()

    target_sources(raytracer_bench PRIVATE render/embree/embree_backend.cpp)
    target_include_directories(raytracer_bench PRIVATE "${EMBREE_INCLUDE_DIR}")
    target_link_libraries(raytracer_bench PRIVATE "${EMBREE_LIBRARY}")
    target_compile_definitions(raytracer_bench PRIVATE
        RT_ENABLE_EMBREE=1
        RT_EMBREE_VERSION_MAJOR=${RT_EMBREE_VERSION_MAJOR})
endif()

if(RT_ENABLE_OPTIX)
    set(OPTIX_ROOT_DIR "" CACHE PATH "Root directory of the NVIDIA OptiX SDK")
    find_path(OPTIX_INCLUDE_DIR
        NAMES optix.h
        HINTS "${OPTIX_ROOT_DIR}" "$ENV{OPTIX_ROOT_DIR}" "$ENV{OPTIX_ROOT}"
        PATH_SUFFIXES include)
    if(NOT OPTIX_INCLUDE_DIR)
        message(FATAL_ERROR
            "RT_ENABLE_OPTIX=ON requires the NVIDIA OptiX SDK; set OPTIX_ROOT_DIR to its installation directory")
    endif()

    foreach(required_optix_header IN ITEMS
            optix_stubs.h
            optix_function_table_definition.h
            optix_device.h
            optix_types.h)
        if(NOT EXISTS "${OPTIX_INCLUDE_DIR}/${required_optix_header}")
            message(FATAL_ERROR
                "OptiX SDK header ${required_optix_header} was not found in ${OPTIX_INCLUDE_DIR}")
        endif()
    endforeach()

    file(STRINGS "${OPTIX_INCLUDE_DIR}/optix.h" RT_OPTIX_VERSION_LINE
        REGEX "^#define[ \t]+OPTIX_VERSION[ \t]+[0-9]+")
    if(NOT RT_OPTIX_VERSION_LINE)
        message(FATAL_ERROR "Could not determine OPTIX_VERSION from ${OPTIX_INCLUDE_DIR}/optix.h")
    endif()
    string(REGEX MATCH "[0-9]+$" RT_OPTIX_VERSION "${RT_OPTIX_VERSION_LINE}")
    if(RT_OPTIX_VERSION LESS 70700)
        message(FATAL_ERROR
            "This backend requires OptiX SDK 7.7 or newer (found OPTIX_VERSION=${RT_OPTIX_VERSION})")
    endif()

    if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL 13.0)
        set(RT_OPTIX_DEFAULT_PTX_ARCH "compute_75")
    else()
        set(RT_OPTIX_DEFAULT_PTX_ARCH "compute_50")
    endif()
    set(RT_OPTIX_PTX_ARCH "${RT_OPTIX_DEFAULT_PTX_ARCH}" CACHE STRING
        "Virtual CUDA architecture used to compile the OptiX device PTX")
    if(NOT RT_OPTIX_PTX_ARCH MATCHES "^compute_[0-9]+$")
        message(FATAL_ERROR "RT_OPTIX_PTX_ARCH must use the form compute_XX (for example compute_75)")
    endif()

    set(RT_OPTIX_PTX "${CMAKE_CURRENT_BINARY_DIR}/optix_programs.ptx")
    add_custom_command(
        OUTPUT "${RT_OPTIX_PTX}"
        COMMAND "${CMAKE_CUDA_COMPILER}" --ptx --std=c++17 --gpu-architecture=${RT_OPTIX_PTX_ARCH}
            -I"${OPTIX_INCLUDE_DIR}"
            -I"${CMAKE_CURRENT_SOURCE_DIR}/render/optix"
            -o "${RT_OPTIX_PTX}"
            "${CMAKE_CURRENT_SOURCE_DIR}/render/optix/optix_programs.cu"
        DEPENDS
            render/optix/optix_programs.cu
            render/optix/optix_shared.hpp
        VERBATIM)
    add_custom_target(raytracer_optix_ptx DEPENDS "${RT_OPTIX_PTX}")

    target_sources(raytracer_bench PRIVATE render/optix/optix_backend.cpp)
    target_include_directories(raytracer_bench PRIVATE "${OPTIX_INCLUDE_DIR}")
    target_link_libraries(raytracer_bench PRIVATE CUDA::cudart CUDA::cuda_driver)
    target_compile_definitions(raytracer_bench PRIVATE
        RT_ENABLE_OPTIX=1
        RT_OPTIX_PTX_PATH=\"${RT_OPTIX_PTX}\")
    add_dependencies(raytracer_bench raytracer_optix_ptx)
endif()

if(MSVC)
    target_compile_options(raytracer_bench PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/O2>
        $<$<COMPILE_LANGUAGE:CXX>:/W4>
        $<$<COMPILE_LANGUAGE:CUDA>:-O3>)
else()
    target_compile_options(raytracer_bench PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-O3>
        $<$<COMPILE_LANGUAGE:CXX>:-Wall>
        $<$<COMPILE_LANGUAGE:CXX>:-Wextra>
        $<$<COMPILE_LANGUAGE:CUDA>:-O3>)
endif()
