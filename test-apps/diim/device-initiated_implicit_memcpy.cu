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

__global__ void simple_kernel(int *src, int *dst1, int *dst2){
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx % 2 == 0) {
    dst1[idx] = modify_cell(src[idx]);
  }
  else {
    dst2[idx] = modify_cell(src[idx]);
  }
}

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
  gpuErrchk(cudaMalloc(&g0, buf_size));

  int *g1 = NULL;

  cudaSetDevice(gpuid[1]);
  gpuErrchk(cudaMalloc(&g1, buf_size));
  cudaSetDevice(gpuid[0]);

  int *g2 = NULL;

  cudaSetDevice(gpuid[2]);
  gpuErrchk(cudaMalloc(&g2, buf_size));
  cudaSetDevice(gpuid[0]);

  int *h0 = NULL;
  gpuErrchk(cudaMallocHost(&h0, buf_size));

  int *h1 = NULL;
  gpuErrchk(cudaMallocHost(&h1, buf_size));

  int *h2 = NULL;
  gpuErrchk(cudaMallocHost(&h2, buf_size));

  gpuErrchk(cudaMemcpy(g0, h0, buf_size, cudaMemcpyHostToDevice));

  cudaSetDevice(gpuid[0]);
  simple_kernel<<<1, size>>>(g0, g1, g2);
  gpuErrchk(cudaMemcpy(h1, g1, buf_size, cudaMemcpyDeviceToHost));
  gpuErrchk(cudaMemcpy(h2, g2, buf_size, cudaMemcpyDeviceToHost));

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
