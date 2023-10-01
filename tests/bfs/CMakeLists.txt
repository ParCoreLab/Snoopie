add_executable(bfs
        bfs_basic.cu)

target_link_libraries(bfs
        CUDA::cudart
        OpenMP::OpenMP_CXX)
