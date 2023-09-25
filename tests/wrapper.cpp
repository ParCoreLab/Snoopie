#include <cuda_runtime.h>
// #include <unistd.h>
#include <cstdint>

// extern "C"
cudaError_t cudaMallocWrap(void **devPtr, size_t size, const char *var_name,
                           const uint32_t element_size,
                           const char *fname, const char *fxname, int lineno /*, const std::experimental::source_location& location = std::experimental::source_location::current()*/) {
  return cudaSuccess;
}

cudaError_t cudaMallocHostWrap(void **devPtr, size_t size, const char *var_name,
                               const uint32_t element_size,
                               const char *fname, const char *fxname, int lineno /*, const std::experimental::source_location& location = std::experimental::source_location::current()*/) {
  return cudaSuccess;
}

void *nvshmem_mallocWrap(size_t size, const char *var_name,
                         const uint32_t element_size, const char *fname,
                         const char *fxname, int lineno) {
  return NULL;
}

void *nvshmem_alignWrap(size_t alignment, size_t size, const char *var_name,
                        const uint32_t element_size, const char *fname,
                        const char *fxname, int lineno) {
  return NULL;
}
