find_package(ZSTD REQUIRED)
find_package(CPPTRACE REQUIRED)

add_library(mem_multigpu SHARED
        mem_multigpu.cu
        inject_funcs.cu
        obj_database.cu
        common.h
        util.h)

target_link_libraries(mem_multigpu
        CUDA::cudart
        cuda
        zstd
        cpptrace::cpptrace
        nvbit::nvbit)

target_include_directories(mem_multigpu SYSTEM PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include")

set_source_files_properties(inject_funcs.cu PROPERTIES COMPILE_FLAGS -maxrregcount=24 -Xptxas -astoolspatch)

target_compile_options(mem_multigpu PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
        -D_FORCE_INLINES
        >)
