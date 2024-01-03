find_package(ZSTD REQUIRED)


add_library(mem_multigpu SHARED
  mem_multigpu.cu
  inject_funcs.cu
  obj_database.cu
        )

set_target_properties(mem_multigpu PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(mem_multigpu PROPERTIES CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(mem_multigpu
        cuda
        zstd
        cpptrace
        nvbit::nvbit
	-L/home/msasongko17/anaconda3/envs/my_env/lib/python3.6/config-3.6m-x86_64-linux-gnu -L/home/msasongko17/anaconda3/envs/my_env/lib -lpython3.6m
	-fno-lto
	-lutil
	)


get_target_property(library_includes cpptrace INTERFACE_INCLUDE_DIRECTORIES)

target_include_directories(mem_multigpu SYSTEM PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_include_directories(mem_multigpu PRIVATE ${library_includes})

set_source_files_properties(inject_funcs.cu PROPERTIES COMPILE_FLAGS -maxrregcount=24 -Xptxas=-astoolspatch --keep-device-functions)

target_compile_options(mem_multigpu PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
      -allow-unsupported-compiler
      --keep-device-functions
      -I/home/msasongko17/anaconda3/envs/my_env/include/python3.6m -I/home/msasongko17/.local/include/python3.6m -I/home/msasongko17/anaconda3/envs/my_env/lib/python3.6/site-packages/numpy/core/include/numpy
      >)

target_link_options(mem_multigpu PRIVATE $<DEVICE_LINK:
    -allow-unsupported-compiler
      -Xptxas=-astoolspatch
      --keep-device-functions
      >)

set(CMAKE_VERBOSE_MAKEFILE ON)
