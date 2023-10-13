#include "cuda_wrapper.hpp"
#include <stdio.h>

//#include <experimental/source_location>
// cudaError_t cudaMalloc ( void** devPtr, size_t size, const
// std::experimental::source_location& location =
// std::experimental::source_location::current());
#define cudaMallocWRAP(a, b, var_name, element_size)                           \
  cudaMallocWrap((void **)a, b, (char *)var_name, element_size, __FILE__,      \
                 __func__, __LINE__)

int main() {

  int *d_a[4], *d_b[6];
  for (int i = 0; i < 4; i++)
    cudaMallocWRAP(&d_a[i], 2 * sizeof(*d_a[0]), "d_a", 4);
  for (int i = 0; i < 6; i++)
    cudaMallocWRAP(&d_b[i], 4 * sizeof(*d_b[0]), "d_b", 4);
  for (int i = 0; i < 4; i++)
    cudaFree(&d_a[i]);
  for (int i = 0; i < 6; i++)
    cudaFree(&d_b[i]);
  return 0;
}
