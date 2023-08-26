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

__device__ int a = 0;

__global__ void simple_kernel1(int *gpu_id){
	a += *gpu_id;
	printf("a: %d in gpu %d\n", a, *gpu_id);
}

__global__ void simple_kernel(int *src, int *dst1, int *dst2){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int b = 10;
  if (idx % 2 == 0) {
    dst1[idx] = modify_cell(src[idx]);
    //dst1[idx] += a;
    //dst1[idx] += dst2[idx];
  }
  else {
    dst2[idx] = modify_cell(src[idx]);
    //dst2[idx] += b;
    //dst2[idx] += dst1[idx];
  }
#if 0
  idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx % 2 == 0) {
    dst1[idx] += modify_cell(src[idx]);
  }
  else {
    dst2[idx] += modify_cell(src[idx]);
  }
#endif
  //printf("hello\n");
}

#if 0
__global__ void simple_kernel1(int *src, int *dst1, int *dst2){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx % 2 == 0) {
    dst1[idx] = modify_cell(src[idx]);
  }
  else {
    dst2[idx] = modify_cell(src[idx]);
  }
  idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx % 2 == 0) {
    dst1[idx] = modify_cell(src[idx]);
  }
  else {
    dst2[idx] = modify_cell(src[idx]);
  }
}
#endif

int main() {
  int gpuid[] = {0, 1, 2};

  int canAccessPeer;
  cudaError_t err = cudaDeviceCanAccessPeer(&canAccessPeer, 0, 1);
  if (err != cudaSuccess) {
  }
  gpuErrchk(cudaDeviceEnablePeerAccess(1, 0));
  gpuErrchk(cudaDeviceEnablePeerAccess(2, 0));
  gpuErrchk(cudaDeviceEnablePeerAccess(3, 0));

  const size_t size = 32;
  const size_t buf_size = size * sizeof(int);

  int *g0 = NULL;
  cudaSetDevice(gpuid[0]);
  gpuErrchk(cudaMallocWRAP(&g0, buf_size, "g0", 4));

  int *g1 = NULL;

  cudaSetDevice(gpuid[1]);
  gpuErrchk(cudaMallocWRAP(&g1, buf_size, "g1", 4));
  cudaSetDevice(gpuid[0]);

  int *g2 = NULL;

  cudaSetDevice(gpuid[2]);
  gpuErrchk(cudaMallocWRAP(&g2, buf_size, "g2", 4));
  cudaSetDevice(gpuid[0]);

  int *h0 = NULL;
  gpuErrchk(cudaMallocHostWRAP(&h0, buf_size, "h0", 4));

  int *h1 = NULL;
  gpuErrchk(cudaMallocHostWRAP(&h1, buf_size, "h1", 4));

  int *h2 = NULL;
  gpuErrchk(cudaMallocHostWRAP(&h2, buf_size, "h2", 4));

  gpuErrchk(cudaMemcpy(g0, h0, buf_size, cudaMemcpyHostToDevice));

  cudaSetDevice(gpuid[0]);
  simple_kernel<<<1, size>>>(g0, g1, g2);

  int gpu_id = 0;
  int gpu_id1 = 1;
  //gpuErrchk(cudaMallocHostWRAP(&gpu_id, 1, "gpu_id", 4));
  //gpuErrchk(cudaMallocHostWRAP(&gpu_id1, 1, "gpu_id", 4));
  //*gpu_id = 0;
  int *gpu_id_dev = NULL;
  int *gpu_id_dev1 = NULL;
  gpuErrchk(cudaMallocWRAP(&gpu_id_dev, 4, "gpu_id_dev", 4)); 
  gpuErrchk(cudaMemcpy(gpu_id_dev, &gpu_id, 4, cudaMemcpyHostToDevice));
  simple_kernel1<<<1, 1>>>(gpu_id_dev);
  cudaSetDevice(gpuid[1]);
  //*gpu_id1 = 1;
  gpuErrchk(cudaMallocWRAP(&gpu_id_dev1, 4, "gpu_id_dev1", 4));
  gpuErrchk(cudaMemcpy(gpu_id_dev1, &gpu_id1, 4, cudaMemcpyHostToDevice));
  simple_kernel1<<<1, 1>>>(gpu_id_dev1);
  cudaSetDevice(gpuid[0]);
  //*gpu_id = gpuid[0];
  //gpuErrchk(cudaMemcpy(gpu_id_dev, gpu_id, 4, cudaMemcpyHostToDevice));
  simple_kernel1<<<1, 1>>>(gpu_id_dev);
  cudaSetDevice(gpuid[1]);
  //*gpu_id = gpuid[1];
  //gpuErrchk(cudaMemcpy(gpu_id_dev1, gpu_id, 4, cudaMemcpyHostToDevice));
  simple_kernel1<<<1, 1>>>(gpu_id_dev1);
  cudaSetDevice(gpuid[0]);
  //*gpu_id = gpuid[0];
  //gpuErrchk(cudaMemcpy(gpu_id_dev, gpu_id, 4, cudaMemcpyHostToDevice));
  simple_kernel1<<<1, 1>>>(gpu_id_dev);
  cudaSetDevice(gpuid[1]);
  //*gpu_id = gpuid[1];
  //gpuErrchk(cudaMemcpy(gpu_id_dev1, gpu_id, 4, cudaMemcpyHostToDevice));
  simple_kernel1<<<1, 1>>>(gpu_id_dev1);
  gpuErrchk(cudaMemcpy(h1, g1, buf_size, cudaMemcpyDeviceToHost));
  gpuErrchk(cudaMemcpy(h2, g2, buf_size, cudaMemcpyDeviceToHost));
  //simple_kernel1<<<1, size>>>(g0, g1, g2);

  for (int i = 0; i < size; i++) {
    printf("\rchecking correctness against CPU: %.2f", ((float) i / (float) size) * 100);
	  if (i % 2 == 0 && h1[i] == modify_cell(h0[i])) {
		  continue;
    } else if (i % 2 == 1 && h2[i] == modify_cell(h0[i])) {
		  continue;
    }

    cout << "FAILED: modify_cell((H0: " << i << ")) " << modify_cell(h0[i]) << "  != (H1: " << i << ") " << h1[i] << endl;
    return 1;
  }

  printf("\ntransfer finished successfully\n");

  
  cudaFree(h0);
  cudaFree(h1);
  cudaFree(h2);
  cudaFree(g0);
  cudaFree(g1);
  cudaFree(g2);

  return 0;
}
