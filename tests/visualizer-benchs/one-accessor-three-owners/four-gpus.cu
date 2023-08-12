#include <iostream>
#include <stdio.h>
#include<unistd.h>

#include "cuda_wrapper.hpp"

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

__global__ void simple_kernel(int *dst, int *src1, int *src2, int *src3){
  //int idx = blockIdx.x * blockDim.x + threadIdx.x;
  dst[50] = modify_cell(src1[50]);
  dst[50] += modify_cell(src2[50]);
  dst[50] += modify_cell(src3[50]);
}

int main() {
  int gpuid[] = {0, 1, 2, 3};

  int canAccessPeer;
  cudaError_t err = cudaDeviceCanAccessPeer(&canAccessPeer, 0, 1);
  if (err != cudaSuccess) {
  }
 
  cudaSetDevice(gpuid[0]);
  gpuErrchk(cudaDeviceEnablePeerAccess(1, 0));
  gpuErrchk(cudaDeviceEnablePeerAccess(2, 0));
  gpuErrchk(cudaDeviceEnablePeerAccess(3, 0));
   

  const size_t size = 100;
  const size_t buf_size = size * sizeof(int);

  int *g0 = NULL;
  int *g1 = NULL;
  int *g2 = NULL;
  int *g3 = NULL;
  int *h0 = NULL;
  int *h1 = NULL;
  int *h2 = NULL;
  int *h3 = NULL;

  //cudaSetDevice(gpuid[0]);
  gpuErrchk(cudaMallocWRAP(&g0, buf_size, "g0", 4));
  gpuErrchk(cudaMallocHostWRAP(&h0, buf_size, "h0", 4));

  cudaSetDevice(gpuid[1]);
  gpuErrchk(cudaMallocWRAP(&g1, buf_size, "g1", 4));
  gpuErrchk(cudaMallocHostWRAP(&h1, buf_size, "h1", 4));
  gpuErrchk(cudaMemcpy(g1, h1, buf_size, cudaMemcpyHostToDevice));

  cudaSetDevice(gpuid[2]);
  gpuErrchk(cudaMallocWRAP(&g2, buf_size, "g2", 4));
  gpuErrchk(cudaMallocHostWRAP(&h2, buf_size, "h2", 4));
  gpuErrchk(cudaMemcpy(g2, h2, buf_size, cudaMemcpyHostToDevice));

  cudaSetDevice(gpuid[3]);
  gpuErrchk(cudaMallocWRAP(&g3, buf_size, "g3", 4));
  gpuErrchk(cudaMallocHostWRAP(&h3, buf_size, "h3", 4));
  gpuErrchk(cudaMemcpy(g3, h3, buf_size, cudaMemcpyHostToDevice));

  cudaSetDevice(gpuid[0]);

  simple_kernel<<<1, size>>>(g0, g1, g2, g3);
  
  gpuErrchk(cudaMemcpy(h0, g0, buf_size, cudaMemcpyDeviceToHost));

  cudaFree(h0);
  cudaFree(h1);
  cudaFree(h2);
  cudaFree(h3);
  cudaFree(g0);
  cudaFree(g1);
  cudaFree(g2);
  cudaFree(g3);

  return 0;
}
