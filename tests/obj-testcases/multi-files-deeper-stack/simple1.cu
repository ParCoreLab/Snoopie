extern __host__ __device__ int modify_cell(int a);

__global__ void simple_kernel1(int size, int *src, int *dst1, int *dst2,
                               int *dst3) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size) {
    return;
  }

  if (idx % 3 == 0) {
    dst2[idx] = modify_cell(src[idx]);
  } else if (idx % 3 == 1) {
    dst3[idx] = modify_cell(src[idx]);
  } else {
    dst1[idx] = modify_cell(src[idx]);
  }
}

cudaError_t foo_alloc(int **buff, size_t size) {
  return cudaMalloc(buff, size);
}

cudaError_t foo_hostalloc(int **buff, size_t size) {
  return cudaMallocHost(buff, size);
}
