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

#set(NVSHMEM_DIR "./cmake/nvshmem")

#find_package(NVSHMEM REQUIRED)
find_package(MPI REQUIRED)
find_package(NVBIT REQUIRED)
find_package(Python REQUIRED)

find_package(OpenMP REQUIRED)
list(APPEND CMAKE_CUDA_FLAGS "-Xcompiler -fopenmp")


add_subdirectory(src/mem_multigpu)
add_subdirectory(tests)

add_subdirectory(tests/bfs)
add_subdirectory(tests/stencil/stencil-nvshmem_nvidia)
add_subdirectory(tests/stencil/stencil-p2p_base)
add_subdirectory(tests/stencil/stencil-tb-singlestream)
add_subdirectory(tests/obj-testcases/nvshmem-multi-files)
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
#  OUTPUT_FILE memop_to_line.txt)

#execute_process(COMMAND cp memop_to_line.txt ${CMAKE_CURRENT_LIST_DIR}/build/tests/stencil/stencil-nvshmem_nvidia
#  WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/tests/stencil/stencil-nvshmem_nvidia)

#add_custom_target(jacobi_object
#  WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/tests/stencil/stencil-nvshmem_nvidia
#  COMMAND nvcc -dc -Xcompiler -fopenmp -gencode arch=compute_80,code=sm_80 -gencode arch=compute_80,code=compute_80 -std=c++17 -ccbin=mpicxx -I${NVSHMEM_HOME}/include -I${MPI_HOME}/include -o jacobi.o jacobi.cu -lineinfo)

#add_custom_target(jacobi_cubin
#  WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/tests/stencil/stencil-nvshmem_nvidia
#  DEPENDS jacobi_object
#  COMMAND cuobjdump jacobi.o -xelf all)

add_custom_target(nvshmem-multi-files_test_mapping
  COMMAND nvcc -cubin -rdc=true -lineinfo device-initiated-direct-access.cu
  COMMAND nvdisasm --print-line-info device-initiated-direct-access.cubin > memop_to_line.txt
  COMMAND nvcc -cubin -rdc=true -lineinfo simple1.cu
  COMMAND nvdisasm --print-line-info simple1.cubin >> memop_to_line.txt
  COMMAND nvcc -cubin -rdc=true -lineinfo simple2.cu
  COMMAND nvdisasm --print-line-info simple2.cubin >> memop_to_line.txt
  WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/tests/obj-testcases/nvshmem-multi-files)

add_custom_target(nvshmem-multi-files_test
  WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/build/tests/obj-testcases/nvshmem-multi-files
  DEPENDS nvshmem-multi-files_test_mapping
  COMMAND mv ${CMAKE_CURRENT_LIST_DIR}/tests/obj-testcases/nvshmem-multi-files/memop_to_line.txt ${CMAKE_CURRENT_LIST_DIR}/build/tests/obj-testcases/nvshmem-multi-files
  COMMAND rm ${CMAKE_CURRENT_LIST_DIR}/tests/obj-testcases/nvshmem-multi-files/*.cubin
  COMMAND ${MPIRUN} --oversubscribe -x NVSHMEM_NGPUS=2 -x NOBANNER=1 -x LD_PRELOAD="../../../src/mem_multigpu/libmem_multigpu.so" -x KERNEL_NAME=all -x CODE_ATTRIBUTION=1 -np ${NP} ./nvshmem-multi-files -n 32
)

add_custom_target(bfs_test_mapping
  COMMAND cuobjdump ./bfs -xelf all
  #COMMAND nvcc -cubin -DHAVE_CUB -Xcompiler -fopenmp -lineinfo -DUSE_NVTX -lnvToolsExt -std=c++14 bfs_basic.cu
  COMMAND nvdisasm --print-line-info bfs*.cubin > memop_to_line.txt
  WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/build/tests/bfs)

add_custom_target(bfs_test
  WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/build/tests/bfs
  DEPENDS bfs_test_mapping
  #COMMAND mv ${CMAKE_CURRENT_LIST_DIR}/tests/bfs/memop_to_line.txt ${CMAKE_CURRENT_LIST_DIR}/build/tests/bfs
  COMMAND rm ${CMAKE_CURRENT_LIST_DIR}/build/tests/bfs/*.cubin
  COMMAND if [ ! -d "gupta2" ] \; then wget https://suitesparse-collection-website.herokuapp.com/MM/Gupta/gupta2.tar.gz && tar -xf gupta2.tar.gz \; fi \;
  #COMMAND tar -xf gupta2.tar.gz
  COMMAND echo "native run"
  COMMAND ${CMAKE_COMMAND} -E env LD_LIBRARY_PATH=.:$$LD_LIBRARY_PATH ./bfs -f gupta2/gupta2.mtx -n 2 -b 10 -t 512 -v 0 -m 0 -r 1 -o 1
  COMMAND echo "profiled run"
  COMMAND ${CMAKE_COMMAND} -E env LD_LIBRARY_PATH=.:$$LD_LIBRARY_PATH LD_PRELOAD=../../src/mem_multigpu/libmem_multigpu.so KERNEL_NAME="all" CODE_ATTRIBUTION=1 ./bfs -f gupta2/gupta2.mtx -n 2 -b 10 -t 512 -v 0 -m 0 -r 1 -o 1
)

add_custom_target(stencil_nvshmem_nvidia_test_mapping
        COMMAND cuobjdump ./stencil-cmake-nvidia -xelf all
        #COMMAND nvcc -dc -Xcompiler -fopenmp -gencode arch=compute_80,code=sm_80 -gencode arch=compute_80,code=compute_80 -std=c++17 -ccbin=mpicxx -I$ENV{NVSHMEM_HOME}/include -I$ENV{MPI_HOME}/include -o jacobi.o jacobi.cu -lineinfo
        #COMMAND cuobjdump jacobi.o -xelf all
  COMMAND nvdisasm --print-line-info stencil-cmake-nvidia*.cubin > memop_to_line.txt
  WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/build/tests/stencil/stencil-nvshmem_nvidia)

add_custom_target(stencil_nvshmem_nvidia_test
  WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/build/tests/stencil/stencil-nvshmem_nvidia
  DEPENDS stencil_nvshmem_nvidia_test_mapping
  #COMMAND mv ${CMAKE_CURRENT_LIST_DIR}/tests/stencil/stencil-nvshmem_nvidia/memop_to_line.txt ${CMAKE_CURRENT_LIST_DIR}/build/tests/stencil/stencil-nvshmem_nvidia
  COMMAND rm ${CMAKE_CURRENT_LIST_DIR}/build/tests/stencil/stencil-nvshmem_nvidia/*.cubin
  COMMAND ${MPIRUN} --oversubscribe -x NVSHMEM_NGPUS="2" -x KERNEL_NAME=all -x LD_LIBRARY_PATH=/usr/local/cuda/lib64:.:$ENV{NVSHMEM_HOME}/lib:$$LD_LIBRARY_PATH -x LD_PRELOAD="../../../src/mem_multigpu/libmem_multigpu.so" -x CODE_ATTRIBUTION=1 -np ${NP} ./stencil-cmake-nvidia -nx ${SIZE} -ny ${SIZE} -niter ${NITER}
)

add_custom_target(stencil_p2p_base_test_mapping
  COMMAND cuobjdump ./stencil-p2p_base -xelf all
#COMMAND nvcc -cubin -DHAVE_CUB -Xcompiler -fopenmp -lineinfo -DUSE_NVTX -lnvToolsExt  -std=c++14 jacobi.cu
  COMMAND nvdisasm --print-line-info stencil-p2p_base*.cubin > memop_to_line.txt
  WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/build/tests/stencil/stencil-p2p_base)

add_custom_target(stencil_p2p_base_test
#WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/build/tests/stencil/stencil-p2p_base
  WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/build/tests/stencil/stencil-p2p_base
  DEPENDS stencil_p2p_base_test_mapping
#COMMAND mv ${CMAKE_CURRENT_LIST_DIR}/tests/stencil/stencil-p2p_base/memop_to_line.txt ${CMAKE_CURRENT_LIST_DIR}/build/tests/stencil/stencil-p2p_base
  COMMAND rm ${CMAKE_CURRENT_LIST_DIR}/build/tests/stencil/stencil-p2p_base/*.cubin
  COMMAND ${CMAKE_COMMAND} -E env NOBANNER=1 KERNEL_NAME=all CODE_ATTRIBUTION=1 LD_LIBRARY_PATH=.:$$LD_LIBRARY_PATH LD_PRELOAD="../../../src/mem_multigpu/libmem_multigpu.so" ./stencil-p2p_base -nx ${SIZE} -ny ${SIZE} -niter ${NITER}
)

add_custom_target(stencil_tb_singlestream_test_mapping
  COMMAND cuobjdump ./stencil-tb_singlestream -xelf all
#COMMAND nvcc -cubin -DHAVE_CUB -Xcompiler -fopenmp -lineinfo -DUSE_NVTX -lnvToolsExt  -std=c++14 jacobi.cu
  COMMAND nvdisasm --print-line-info stencil-tb_singlestream*.cubin > memop_to_line.txt
  WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/build/tests/stencil/stencil-tb-singlestream)

add_custom_target(stencil_tb_singlestream_test
  WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/build/tests/stencil/stencil-tb-singlestream
  DEPENDS stencil_tb_singlestream_test_mapping
  #COMMAND mv ${CMAKE_CURRENT_LIST_DIR}/tests/stencil/stencil-tb-singlestream/memop_to_line.txt ${CMAKE_CURRENT_LIST_DIR}/build/tests/stencil/stencil-tb-singlestream
  COMMAND rm ${CMAKE_CURRENT_LIST_DIR}/build/tests/stencil/stencil-tb-singlestream/*.cubin
  COMMAND ${CMAKE_COMMAND} -E env NOBANNER=1 KERNEL_NAME=all CODE_ATTRIBUTION=1 LD_LIBRARY_PATH=.:$$LD_LIBRARY_PATH LD_PRELOAD="../../../src/mem_multigpu/libmem_multigpu.so" ./stencil-tb_singlestream -nx ${SIZE} -ny ${SIZE} -niter ${NITER}
)
#add_dependencies(stencil_nvshmem_nvidia_test ${CMAKE_CURRENT_LIST_DIR}/tests/stencil/stencil-nvshmem_nvidia/jacobi_mapping)
