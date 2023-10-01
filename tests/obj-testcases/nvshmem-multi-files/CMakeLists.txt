find_package(OpenMP)

add_executable(nvshmem-multi-files
        device-initiated-direct-access.cu simple1.cu simple2.cu)

target_include_directories(nvshmem-multi-files PRIVATE $ENV{NVSHMEM_HOME}/include)

message("$ENV{NVSHMEM_HOME}/include")

target_link_libraries(nvshmem-multi-files
	-L$ENV{NVSHMEM_HOME}/lib -lnvshmem_host -lnvshmem_device
        mem_multigpu
        CUDA::cudart
        cuda
        MPI::MPI_CXX
        OpenMP::OpenMP_CXX)
