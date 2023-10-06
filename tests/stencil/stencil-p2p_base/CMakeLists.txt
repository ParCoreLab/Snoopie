add_executable(stencil-p2p_base
        jacobi.cu)

target_compile_options(stencil-p2p_base PRIVATE
      -g
      -lineinfo)

target_link_libraries(stencil-p2p_base
        wrapper
        mem_multigpu
        CUDA::cudart
        OpenMP::OpenMP_CXX)
