find_package(ZSTD REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development NumPy)
#find_package(pybind11 REQUIRED)

add_library(mem_multigpu SHARED
  mem_multigpu.cu
  inject_funcs.cu
  obj_database.cu
        )

set_target_properties(mem_multigpu PROPERTIES OUTPUT_NAME "mem_multigpu")

set_target_properties(mem_multigpu PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(mem_multigpu PROPERTIES CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(mem_multigpu
        cuda
        zstd
        cpptrace
        nvbit::nvbit
	Python3::NumPy
	Python3::Python
	-fno-lto
	-lutil
	)

target_include_directories(mem_multigpu SYSTEM PRIVATE "${CMAKE_BINARY_DIR}/_deps/pybind11-src/include")

get_target_property(library_includes cpptrace INTERFACE_INCLUDE_DIRECTORIES)
#get_target_property(library_includes pybind11 INTERFACE_INCLUDE_DIRECTORIES)

target_include_directories(mem_multigpu SYSTEM PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include")

set_source_files_properties(inject_funcs.cu PROPERTIES COMPILE_FLAGS -maxrregcount=24 -Xptxas=-astoolspatch --keep-device-functions)

target_compile_options(mem_multigpu PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
      -allow-unsupported-compiler
      --keep-device-functions
      #-I${CMAKE_SOURCE_DIR}/extern/pybind11/include
      #-I/home/it4i-msasong/.conda/envs/dstest/include
      #-I/home/msasongko17/anaconda3/envs/my_env/include/python3.6m -I/home/msasongko17/.local/include/python3.6m -I/home/msasongko17/anaconda3/envs/my_env/lib/python3.6/site-packages/numpy/core/include/numpy
      >)

target_link_options(mem_multigpu PRIVATE $<DEVICE_LINK:
    -allow-unsupported-compiler
      -Xptxas=-astoolspatch
      --keep-device-functions
      >)

#pybind11_extension(mem_multigpu)

set(CMAKE_VERBOSE_MAKEFILE ON)
