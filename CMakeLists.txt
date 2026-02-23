cmake_minimum_required(VERSION 3.15...3.27)
project(pytafast VERSION 0.1.0)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Build ta-lib
# ta-lib provides a configure script, but CMake can build it directly if we list its sources.
# Since ta-lib v0.6.4 is already a robust C library, a clean approach is either to use its configure/make (ExternalProject_Add) 
# or just compile its `src` folder directly. Let's use `add_subdirectory` if it has CMake, else we build the sources.
# Let's inspect ta-lib after checkout.

if(WIN32 AND NOT DEFINED ENV{Platform})
    if(CMAKE_SIZEOF_VOID_P EQUAL 8)
        # 64-bit target
        if(CMAKE_SYSTEM_PROCESSOR MATCHES "ARM64|arm64|aarch64")
            set(ENV{Platform} "arm64")
        else()
            set(ENV{Platform} "x64")
        endif()
    else()
        # 32-bit target
        if(CMAKE_SYSTEM_PROCESSOR MATCHES "ARM|arm")
            set(ENV{Platform} "arm")
        else()
            set(ENV{Platform} "x86")
        endif()
    endif()
endif()

add_subdirectory(third_party/ta-lib)

# include nanobind
find_package(Python 3.11 COMPONENTS Interpreter Development.Module REQUIRED)
add_subdirectory(third_party/nanobind)

# The main python module
nanobind_add_module(pytafast_ext src/pytafast_ext.cpp)

# Link against ta-lib
target_link_libraries(pytafast_ext PRIVATE ta-lib-static)

# Install extension in package
install(TARGETS pytafast_ext DESTINATION pytafast)
