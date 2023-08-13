find_package(OpenMP)

add_executable(stencil-cmake-nvidia
        jacobi.cu)

target_link_libraries(stencil-cmake-nvidia
        wrapper
        mem_multigpu
        CUDA::cudart
        MPI::MPI_CXX
        OpenMP::OpenMP_CXX)
