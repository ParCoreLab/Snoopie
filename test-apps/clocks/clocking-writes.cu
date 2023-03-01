#include <iostream>
#include <stdio.h>
#include <unistd.h>
#include <cuda/std/chrono>

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

__global__ void simple_kernel(int *src, int *dst, cuda::std::chrono::system_clock::time_point *clock_res){

  // const int idx = blockIdx.x * blockDim.x + threadIdx.x;

  // Write Locally
  clock_res[0] = cuda::std::chrono::high_resolution_clock::now();
  for (int j = 0; j < 5; j++) {
    for (int i = 0; i < 20000; i++) {
      src[i] = src[i] + 1;
    }
  }
  
  clock_res[1] = cuda::std::chrono::high_resolution_clock::now();

  // Write Remotely
  
  for (int j = 0; j < 5; j++) {
    for (int i = 0; i < 20000; i++) {
      dst[i] = dst[i] + 1;
    }
  }
  clock_res[2] = cuda::std::chrono::high_resolution_clock::now();

}

int main() {
  int gpuid[] = {0, 1};

  cuda::std::chrono::system_clock::time_point *gp_clock_res = NULL;
  cuda::std::chrono::system_clock::time_point *hs_clock_res = NULL;

  cudaSetDevice(gpuid[0]);
  cudaMalloc(&gp_clock_res, 4 * sizeof(cuda::std::chrono::system_clock::time_point));
  hs_clock_res = (cuda::std::chrono::system_clock::time_point *) malloc(4 * sizeof(cuda::std::chrono::system_clock::time_point));

  int canAccessPeer;
  cudaError_t err = cudaDeviceCanAccessPeer(&canAccessPeer, 0, 1);
  if (err != cudaSuccess) {
  }
  gpuErrchk(cudaDeviceEnablePeerAccess(1, 0));
  gpuErrchk(cudaDeviceEnablePeerAccess(2, 0));
  gpuErrchk(cudaDeviceEnablePeerAccess(3, 0));

  const size_t size = 20000;
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
  const dim3 threads(1, 1);
  const dim3 blocks((buf_size / sizeof(int)) / threads.x, 1);

  std::cout << "Launching Kernel" << std::endl;
  cudaSetDevice(gpuid[0]);
  cuda::std::chrono::system_clock::time_point start = cuda::std::chrono::high_resolution_clock::now();
  simple_kernel<<<1, 1>>>(g0, g1, gp_clock_res);
  cudaDeviceSynchronize();
  cuda::std::chrono::system_clock::time_point end = cuda::std::chrono::high_resolution_clock::now();
  gpuErrchk(cudaMemcpy(h1, g1, buf_size, cudaMemcpyDeviceToHost));
  gpuErrchk(cudaMemcpy(hs_clock_res, gp_clock_res, 4 * sizeof(cuda::std::chrono::system_clock::time_point), cudaMemcpyDeviceToHost));
  std::cout << "locale        :"  << cuda::std::chrono::duration_cast<cuda::std::chrono::microseconds>(hs_clock_res[1] - hs_clock_res[0]).count() << std::endl;
  std::cout << "remote        :"  << cuda::std::chrono::duration_cast<cuda::std::chrono::microseconds>(hs_clock_res[2] - hs_clock_res[1]).count() << std::endl;
  std::cout << "remote + local:"  << cuda::std::chrono::duration_cast<cuda::std::chrono::microseconds>(hs_clock_res[2] - hs_clock_res[0]).count() << std::endl;
  std::cout << "Total         :"  << cuda::std::chrono::duration_cast<cuda::std::chrono::microseconds>(end - start).count() << std::endl;

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
  cudaFree(g0);
  cudaFree(g1);

  return 0;
}