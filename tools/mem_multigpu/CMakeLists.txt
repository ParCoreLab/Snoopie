find_package(ZSTD REQUIRED)

add_library(mem_multigpu SHARED
  mem_multigpu.cu
  inject_funcs.cu
  obj_database.cu
        )

target_link_libraries(mem_multigpu
        cuda
        zstd
        cpptrace
        nvbit::nvbit)


# target_include_directories(mem_multigpu PRIVATE )

get_target_property(library_includes cpptrace INTERFACE_INCLUDE_DIRECTORIES)

target_include_directories(mem_multigpu SYSTEM PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_include_directories(mem_multigpu PRIVATE ${library_includes})

set_source_files_properties(inject_funcs.cu PROPERTIES COMPILE_FLAGS -maxrregcount=24 -Xptxas -astoolspatch --keep-device-functions -Xcompiler -Wall -Xcompiler -fPIC -allow-unsupported-compiler)

target_compile_options(mem_multigpu PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
        -D_FORCE_INLINES -lcudadevrt
        >)
