cmake_minimum_required(VERSION 4.0)
project(comparing_nodes_backend)

set(CMAKE_CXX_STANDARD 20)

if(APPLE)
    if(EXISTS "/opt/homebrew/opt/libomp")
        list(APPEND CMAKE_PREFIX_PATH "/opt/homebrew/opt/libomp")
    elseif(EXISTS "/usr/local/opt/libomp")
        list(APPEND CMAKE_PREFIX_PATH "/usr/local/opt/libomp")
    endif()
endif()

find_package(OpenMP REQUIRED)

add_executable(comparing_nodes_backend main.cpp
        csv_loader.hpp
        Solution.cpp
        Solution.hpp
        Cut.hpp
        argparse.hpp
        rng_freezer.hpp
)

include(CheckIPOSupported)
check_ipo_supported(RESULT supported OUTPUT error)
if( supported )
    message(STATUS "IPO / LTO enabled")
    set_property(TARGET comparing_nodes_backend PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
else()
    message(STATUS "IPO / LTO not supported: <${error}>")
endif()

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-march=native" COMPILER_SUPPORTS_MARCH_NATIVE)
if(COMPILER_SUPPORTS_MARCH_NATIVE)
    message(STATUS "Building with -march=native")
    target_compile_options(comparing_nodes_backend PRIVATE -march=native)
endif()

if(OpenMP_CXX_FOUND)
    target_link_libraries(comparing_nodes_backend PUBLIC OpenMP::OpenMP_CXX)
endif()


target_compile_options(comparing_nodes_backend PRIVATE -ffast-math)
target_compile_options(comparing_nodes_backend PRIVATE -O3)
