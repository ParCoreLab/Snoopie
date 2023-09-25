cmake_minimum_required(VERSION 3.23)

project(snoopie VERSION 1.0
        DESCRIPTION "ok"
        LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)

# Minimum supported NVCC version
set(NVCC_VER_MIN 10.1)

# Debugging Makefile
set(CMAKE_VERBOSE_MAKEFILE ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Not portable
add_link_options("-Wl,--no-undefined")

# CUDA
set(CMAKE_CUDA_ARCHITECTURES "80")
find_package(CUDAToolkit ${NVCC_VER_MIN} REQUIRED)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

include(FetchContent)
FetchContent_Declare(
  cpptrace
  GIT_REPOSITORY https://github.com/jeremy-rifkin/cpptrace.git
  GIT_TAG        v0.1.1 # <HASH or TAG>
)
FetchContent_MakeAvailable(cpptrace)

enable_language(CUDA)

set(NVSHMEM_DIR "./cmake/nvshmem")

find_package(NVSHMEM REQUIRED)
find_package(MPI REQUIRED)
find_package(NVBIT REQUIRED)

add_subdirectory(tools/mem_multigpu)
add_subdirectory(tests)

add_subdirectory(tests/stencil/stencil-nvshmem_nvidia)
