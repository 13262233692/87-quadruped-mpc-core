cmake_minimum_required(VERSION 3.16)
project(quadruped_srbd LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

if(MSVC)
    add_compile_options(/W3 /EHsc)
else()
    add_compile_options(-Wall -Wextra)
endif()

find_package(Eigen3 3.3 QUIET)
if(NOT Eigen3_FOUND)
    include(FetchContent)
    FetchContent_Declare(
        eigen
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG 3.4.0
    )
    FetchContent_MakeAvailable(eigen)
endif()

include_directories(${CMAKE_SOURCE_DIR}/include)

add_library(srbd_core STATIC
    src/srbd_model.cpp
    src/bezier_curve.cpp
    src/swing_trajectory.cpp
)

target_link_libraries(srbd_core PUBLIC Eigen3::Eigen)

option(BUILD_PYTHON_BINDINGS "Build Python bindings with pybind11" ON)
if(BUILD_PYTHON_BINDINGS)
    find_package(pybind11 QUIET)
    if(NOT pybind11_FOUND)
        include(FetchContent)
        FetchContent_Declare(
            pybind11
            GIT_REPOSITORY https://github.com/pybind/pybind11.git
            GIT_TAG v2.11.1
        )
        FetchContent_MakeAvailable(pybind11)
    endif()

    pybind11_add_module(quadruped_srbd
        src/py_srbd_bindings.cpp
    )
    target_link_libraries(quadruped_srbd PRIVATE srbd_core)
    set_target_properties(quadruped_srbd PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/python
    )
endif()

option(BUILD_TESTS "Build unit tests" ON)
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
