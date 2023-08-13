find_library(NVBIT_PATH
        NAMES nvbit
        HINTS "${CMAKE_SOURCE_DIR}/core")

add_library(nvbit::nvbit UNKNOWN IMPORTED)

set_target_properties(nvbit::nvbit
        PROPERTIES
        IMPORTED_LOCATION ${NVBIT_PATH}
        INTERFACE_INCLUDE_DIRECTORIES ${CMAKE_SOURCE_DIR}/core)

mark_as_advanced(NVBIT_PATH)