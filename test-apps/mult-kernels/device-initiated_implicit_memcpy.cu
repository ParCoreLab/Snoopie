#include <iostream>
#include <stdio.h>
#include<unistd.h>

using namespace std;


#define gpuErrchk(ans) { gpuAssert(ans); }
inline void gpuAssert(cudaError_t code)
{
  if (code != cudaSuccess) {
  fprintf(stderr,"GPUassert: %s\n", cudaGetErrorString(code));
  }
}

__host__ __device__ int modify_cell(int a) {
  return a + 2;
}

__device__ void simple_kernel111(int *src, int *dst){
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  dst[idx] = modify_cell(src[idx]);
}

__global__ void simple_kernel(int *src, int *dst){
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  dst[idx] = modify_cell(src[idx]);
}
__global__ void simple_kernel(int *src, int *dst, int *another){
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  dst[idx] = modify_cell(src[idx]);
  another[idx] += dst[idx];
}
__global__ void simple_kernel1(int *src, int *dst){
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  dst[idx] = modify_cell(src[idx]);
}

__global__ void simple_kernel2(int *src, int *dst){
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  dst[idx] = modify_cell(src[idx]);
}

__global__ void simple_kernel11(int *src, int *dst){
  simple_kernel111(src, dst);
}

int main() {
  int gpuid[] = {0, 1};

  int canAccessPeer;
  cudaError_t err = cudaDeviceCanAccessPeer(&canAccessPeer, 0, 1);
  if (err != cudaSuccess) {
  }
  gpuErrchk(cudaDeviceEnablePeerAccess(1, 0));

  const size_t size = 32;
  const size_t buf_size = size * sizeof(int);

  int *g0 = NULL;
  cudaSetDevice(gpuid[0]);
  gpuErrchk(cudaMalloc(&g0, buf_size));

  int *g1 = NULL;

  cudaSetDevice(gpuid[1]);
  gpuErrchk(cudaMalloc(&g1, buf_size));
  cudaSetDevice(gpuid[0]);

  int *h0 = NULL;
  gpuErrchk(cudaMallocHost(&h0, buf_size));

  int *h1 = NULL;
  gpuErrchk(cudaMallocHost(&h1, buf_size));


  gpuErrchk(cudaMemcpy(g0, h0, buf_size, cudaMemcpyHostToDevice));
  // gpuErrchk(cudaMemcpy(g1, g0, buf_size, cudaMemcpyDeviceToDevice));
  const dim3 threads(512, 1);
  const dim3 blocks((buf_size / sizeof(int)) / threads.x, 1);

  cudaSetDevice(gpuid[0]);
  // simple_kernel<<<blocks, threads>>>(g0, g1);
  simple_kernel<<<1, 32>>>(g0, g1);
  simple_kernel1<<<1, 32>>>(g0, g1);
  simple_kernel2<<<1, 32>>>(g0, g1);
  simple_kernel11<<<1, 32>>>(g0, g1);
  gpuErrchk(cudaMemcpy(h1, g1, buf_size, cudaMemcpyDeviceToHost));

  for (int i = 0; i < size; i++) {
    printf("\rchecking correctness against CPU: %.2f", ((float) i / (float) size) * 100);
	  if (h1[i] == modify_cell(h0[i])) {
		  continue;
    }

    cout << "FAILED: modify_cell((H0: " << i << ")) " << modify_cell(h0[i]) << "  != (H1: " << i << ") " << h1[i] << endl;
    return 1;
  }

  printf("\ntransfer finished successfully\n");

  
  cudaFree(h0);
  cudaFree(h1);
  cudaFree(g0);
  cudaFree(g1);

  return 0;
}
