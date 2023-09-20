add_library(wrapper SHARED
        wrapper.cpp)

target_link_libraries(wrapper CUDA::cudart)

target_compile_options(mem_multigpu PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
        --generate-line-info
        >)
