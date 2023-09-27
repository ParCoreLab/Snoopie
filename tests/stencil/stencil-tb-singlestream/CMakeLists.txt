add_executable(stencil-tb_singlestream
        jacobi.cu)

target_link_libraries(stencil-tb_singlestream
        mem_multigpu
        CUDA::cudart
        OpenMP::OpenMP_CXX)
