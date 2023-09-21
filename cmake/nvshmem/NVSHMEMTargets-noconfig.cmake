#----------------------------------------------------------------
# Generated CMake target import file.
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "nvshmem::nvshmem" for configuration ""
set_property(TARGET nvshmem::nvshmem APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(nvshmem::nvshmem PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_NOCONFIG "CUDA;CXX"
  IMPORTED_LOCATION_NOCONFIG "${_IMPORT_PREFIX}/lib/libnvshmem.a"
  )

list(APPEND _cmake_import_check_targets nvshmem::nvshmem )
list(APPEND _cmake_import_check_files_for_nvshmem::nvshmem "${_IMPORT_PREFIX}/lib/libnvshmem.a" )

# Import target "nvshmem::nvshmem_host" for configuration ""
set_property(TARGET nvshmem::nvshmem_host APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(nvshmem::nvshmem_host PROPERTIES
  IMPORTED_LOCATION_NOCONFIG "${_IMPORT_PREFIX}/lib/libnvshmem_host.so.2.9.0"
  IMPORTED_SONAME_NOCONFIG "libnvshmem_host.so.2"
  )

list(APPEND _cmake_import_check_targets nvshmem::nvshmem_host )
list(APPEND _cmake_import_check_files_for_nvshmem::nvshmem_host "${_IMPORT_PREFIX}/lib/libnvshmem_host.so.2.9.0" )

# Import target "nvshmem::nvshmem_device" for configuration ""
set_property(TARGET nvshmem::nvshmem_device APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(nvshmem::nvshmem_device PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_NOCONFIG "CUDA"
  IMPORTED_LOCATION_NOCONFIG "${_IMPORT_PREFIX}/lib/libnvshmem_device.a"
  )

list(APPEND _cmake_import_check_targets nvshmem::nvshmem_device )
list(APPEND _cmake_import_check_files_for_nvshmem::nvshmem_device "${_IMPORT_PREFIX}/lib/libnvshmem_device.a" )

# Import target "nvshmem::nvshmem-info" for configuration ""
set_property(TARGET nvshmem::nvshmem-info APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(nvshmem::nvshmem-info PROPERTIES
  IMPORTED_LOCATION_NOCONFIG "${_IMPORT_PREFIX}/bin/nvshmem-info"
  )

list(APPEND _cmake_import_check_targets nvshmem::nvshmem-info )
list(APPEND _cmake_import_check_files_for_nvshmem::nvshmem-info "${_IMPORT_PREFIX}/bin/nvshmem-info" )

# Import target "nvshmem::nvshmem_bootstrap_pmi" for configuration ""
set_property(TARGET nvshmem::nvshmem_bootstrap_pmi APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(nvshmem::nvshmem_bootstrap_pmi PROPERTIES
  IMPORTED_LOCATION_NOCONFIG "${_IMPORT_PREFIX}/lib/nvshmem_bootstrap_pmi.so.2.8.0"
  IMPORTED_SONAME_NOCONFIG "nvshmem_bootstrap_pmi.so.2"
  )

list(APPEND _cmake_import_check_targets nvshmem::nvshmem_bootstrap_pmi )
list(APPEND _cmake_import_check_files_for_nvshmem::nvshmem_bootstrap_pmi "${_IMPORT_PREFIX}/lib/nvshmem_bootstrap_pmi.so.2.8.0" )

# Import target "nvshmem::nvshmem_bootstrap_pmi2" for configuration ""
set_property(TARGET nvshmem::nvshmem_bootstrap_pmi2 APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(nvshmem::nvshmem_bootstrap_pmi2 PROPERTIES
  IMPORTED_LOCATION_NOCONFIG "${_IMPORT_PREFIX}/lib/nvshmem_bootstrap_pmi2.so.2.8.0"
  IMPORTED_SONAME_NOCONFIG "nvshmem_bootstrap_pmi2.so.2"
  )

list(APPEND _cmake_import_check_targets nvshmem::nvshmem_bootstrap_pmi2 )
list(APPEND _cmake_import_check_files_for_nvshmem::nvshmem_bootstrap_pmi2 "${_IMPORT_PREFIX}/lib/nvshmem_bootstrap_pmi2.so.2.8.0" )

# Import target "nvshmem::nvshmem_bootstrap_mpi" for configuration ""
set_property(TARGET nvshmem::nvshmem_bootstrap_mpi APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(nvshmem::nvshmem_bootstrap_mpi PROPERTIES
  IMPORTED_LOCATION_NOCONFIG "${_IMPORT_PREFIX}/lib/nvshmem_bootstrap_mpi.so.2.8.0"
  IMPORTED_SONAME_NOCONFIG "nvshmem_bootstrap_mpi.so.2"
  )

list(APPEND _cmake_import_check_targets nvshmem::nvshmem_bootstrap_mpi )
list(APPEND _cmake_import_check_files_for_nvshmem::nvshmem_bootstrap_mpi "${_IMPORT_PREFIX}/lib/nvshmem_bootstrap_mpi.so.2.8.0" )

# Import target "nvshmem::nvshmem_transport_ibrc" for configuration ""
set_property(TARGET nvshmem::nvshmem_transport_ibrc APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(nvshmem::nvshmem_transport_ibrc PROPERTIES
  IMPORTED_LOCATION_NOCONFIG "${_IMPORT_PREFIX}/lib/nvshmem_transport_ibrc.so.1.0.0"
  IMPORTED_SONAME_NOCONFIG "nvshmem_transport_ibrc.so.1"
  )

list(APPEND _cmake_import_check_targets nvshmem::nvshmem_transport_ibrc )
list(APPEND _cmake_import_check_files_for_nvshmem::nvshmem_transport_ibrc "${_IMPORT_PREFIX}/lib/nvshmem_transport_ibrc.so.1.0.0" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
