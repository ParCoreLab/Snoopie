extern __host__ __device__ int modify_cell(int a);

__global__ void simple_kernel2(int size, int *src, int *dst1, int *dst2,
                               int *dst3) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size) {
    return;
  }

  if (idx % 3 == 0) {
    dst3[idx] = modify_cell(src[idx]);
  } else if (idx % 3 == 1) {
    dst1[idx] = modify_cell(src[idx]);
  } else {
    dst2[idx] = modify_cell(src[idx]);
  }
}
