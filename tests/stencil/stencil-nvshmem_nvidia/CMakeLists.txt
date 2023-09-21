find_package(OpenMP)

add_executable(stencil-cmake-nvidia
        jacobi.cu)

target_link_libraries(stencil-cmake-nvidia
        nvshmem::nvshmem
        wrapper
        mem_multigpu
        CUDA::cudart
        cuda
        MPI::MPI_CXX
        OpenMP::OpenMP_CXX)
