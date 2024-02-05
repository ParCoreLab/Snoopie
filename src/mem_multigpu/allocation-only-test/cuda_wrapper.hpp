#include <cstdint>
cudaError_t cudaMallocWrap(void **devPtr, size_t size, const char *var_name,
                           const uint32_t element_size,
                           const char *fname, const char *fxname, int lineno /*, const std::experimental::source_location& location = std::experimental::source_location::current()*/);
