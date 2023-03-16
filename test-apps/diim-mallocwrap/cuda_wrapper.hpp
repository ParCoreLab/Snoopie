cudaError_t cudaMallocWrap ( void** devPtr, size_t size, const char *var_name, const char *fname, const char *fxname, int lineno/*, const std::experimental::source_location& location = std::experimental::source_location::current()*/);
cudaError_t cudaMallocHostWrap ( void** devPtr, size_t size, const char *var_name, const char *fname, const char *fxname, int lineno);

#define cudaMallocWRAP(a, b, var_name) cudaMallocWrap((void **)a, b, (char *) var_name, __FILE__, __func__, __LINE__)
#define cudaMallocHostWRAP(a, b, var_name) cudaMallocHostWrap((void **)a, b, (char *) var_name, __FILE__, __func__, __LINE__)
