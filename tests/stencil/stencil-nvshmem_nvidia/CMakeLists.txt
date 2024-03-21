find_package(OpenMP)

add_executable(stencil-cmake-nvidia
        jacobi.cu)

target_compile_options(stencil-cmake-nvidia PRIVATE
      -g
      -lineinfo)

target_include_directories(stencil-cmake-nvidia PRIVATE $ENV{NVSHMEM_HOME}/include)

message("$ENV{NVSHMEM_HOME}/include")

target_link_libraries(stencil-cmake-nvidia
	-L$ENV{NVSHMEM_HOME}/lib -lnvshmem_host -lnvshmem_device -lnvidia-ml
        mem_multigpu
        CUDA::cudart
	#        cuda
        MPI::MPI_CXX
        OpenMP::OpenMP_CXX)
