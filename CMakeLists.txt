cmake_minimum_required(VERSION 3.15...3.27)
project(pytalib VERSION 0.1.0)

# Build ta-lib
# ta-lib provides a configure script, but CMake can build it directly if we list its sources.
# Since ta-lib v0.6.4 is already a robust C library, a clean approach is either to use its configure/make (ExternalProject_Add) 
# or just compile its `src` folder directly. Let's use `add_subdirectory` if it has CMake, else we build the sources.
# Let's inspect ta-lib after checkout.
add_subdirectory(third_party/ta-lib)

# include nanobind
find_package(Python 3.8 COMPONENTS Interpreter Development.Module REQUIRED)
add_subdirectory(third_party/nanobind)

# The main python module
nanobind_add_module(pytalib_ext src/pytalib_ext.cpp)

# Link against ta-lib
target_link_libraries(pytalib_ext PRIVATE ta-lib-static)

# Install extension in package
install(TARGETS pytalib_ext DESTINATION pytalib)
