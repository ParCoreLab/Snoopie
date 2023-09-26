cmake_minimum_required(VERSION 3.23)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_ARCHITECTURES 75)

project(snoopie VERSION 1.0
        DESCRIPTION "ok"
        LANGUAGES CXX CUDA)


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

#message("${CMAKE_CURRENT_LIST_DIR}/tests/stencil/stencil-nvshmem_nvidia")

set(NP 2)
set(SIZE 1024)
set(NITER 5)
set(MPIRUN mpirun)

#execute_process(COMMAND nvcc -dc -Xcompiler -fopenmp -gencode arch=compute_80,code=sm_80 -gencode arch=compute_80,code=compute_80 -std=c++17 -ccbin=mpicxx -I${NVSHMEM_HOME}/include -I${MPI_HOME}/include -o jacobi.o jacobi.cu -lineinfo
#  WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/tests/stencil/stencil-nvshmem_nvidia)

#execute_process(COMMAND cuobjdump jacobi.o -xelf all
#  WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/tests/stencil/stencil-nvshmem_nvidia)

#execute_process(COMMAND nvdisasm --print-line-info jacobi.sm_80.cubin
#  WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/tests/stencil/stencil-nvshmem_nvidia
#  OUTPUT_FILE testfile.txt)

#execute_process(COMMAND cp testfile.txt ${CMAKE_CURRENT_LIST_DIR}/build/tests/stencil/stencil-nvshmem_nvidia
#  WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/tests/stencil/stencil-nvshmem_nvidia)

#add_custom_target(jacobi_object
#  WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/tests/stencil/stencil-nvshmem_nvidia
#  COMMAND nvcc -dc -Xcompiler -fopenmp -gencode arch=compute_80,code=sm_80 -gencode arch=compute_80,code=compute_80 -std=c++17 -ccbin=mpicxx -I${NVSHMEM_HOME}/include -I${MPI_HOME}/include -o jacobi.o jacobi.cu -lineinfo)

#add_custom_target(jacobi_cubin
#  WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/tests/stencil/stencil-nvshmem_nvidia
#  DEPENDS jacobi_object
#  COMMAND cuobjdump jacobi.o -xelf all)

add_custom_target(stencil_nvshmem_nvidia_test_mapping 
  COMMAND nvcc -dc -Xcompiler -fopenmp -gencode arch=compute_80,code=sm_80 -gencode arch=compute_80,code=compute_80 -std=c++17 -ccbin=mpicxx -I${NVSHMEM_HOME}/include -I${MPI_HOME}/include -o jacobi.o jacobi.cu -lineinfo
  COMMAND cuobjdump jacobi.o -xelf all
  COMMAND nvdisasm --print-line-info jacobi.sm_80.cubin > testfile.txt
  WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/tests/stencil/stencil-nvshmem_nvidia)

add_custom_target(stencil_nvshmem_nvidia_test
  WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/build/tests/stencil/stencil-nvshmem_nvidia
  DEPENDS stencil_nvshmem_nvidia_test_mapping
  COMMAND mv ${CMAKE_CURRENT_LIST_DIR}/tests/stencil/stencil-nvshmem_nvidia/testfile.txt ${CMAKE_CURRENT_LIST_DIR}/build/tests/stencil/stencil-nvshmem_nvidia
  COMMAND rm ${CMAKE_CURRENT_LIST_DIR}/tests/stencil/stencil-nvshmem_nvidia/*.o ${CMAKE_CURRENT_LIST_DIR}/tests/stencil/stencil-nvshmem_nvidia/*.cubin
  COMMAND ${MPIRUN} --oversubscribe -x NVSHMEM_NGPUS=2 -x KERNEL_NAME="void jacobi_kernel<32, 32>" -x LD_LIBRARY_PATH=/usr/local/cuda/lib64:.:${NVSHMEM_HOME}/lib:$$LD_LIBRARY_PATH -x LD_PRELOAD="../../../tools/mem_multigpu/libmem_multigpu.so" -x CODE_ATTRIBUTION=1 -np ${NP} ./stencil-cmake-nvidia -nx ${SIZE} -ny ${SIZE} -niter ${NITER}
)

#add_dependencies(stencil_nvshmem_nvidia_test ${CMAKE_CURRENT_LIST_DIR}/tests/stencil/stencil-nvshmem_nvidia/jacobi_mapping)  
