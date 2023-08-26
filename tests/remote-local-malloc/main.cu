#include <iostream>
#include <assert.h>
#include <stdio.h>
#include <unistd.h>
#include <cuda/std/chrono>

using namespace std;

// __device__ volatile int *shared_ptr = NULL;

#define gpuErrchk(ans) { gpuAssert(ans); }

__host__ __device__ inline void gpuAssert(cudaError_t code)
{
  if (code != cudaSuccess) {
    printf("GPUassert: %d\n", code);
  }
}

__global__ void alloc_spin(int **shared_ptr){
  gpuErrchk(cudaMalloc(shared_ptr, sizeof(int)));
  printf("*shared ptr is now: %p\n", *shared_ptr);
  **shared_ptr = 10;
}

__global__ void read_spin(int **shared_ptr){
  printf("Attempting to dereference shared ptr -> ");
  printf("*shared_ptr: %p\n", *shared_ptr);
  while(shared_ptr == NULL) { }
  printf("i: %d\n", **shared_ptr);
}


int main() {


  cudaStream_t stream1;
  cudaStreamCreate(&stream1);


  cudaSetDevice(0);
  int **shared_ptr;
  cudaMalloc(&shared_ptr, sizeof(*shared_ptr));
  // set_null<<<1, 1, 1, stream1>>>(shared_ptr);
  // cudaDeviceSynchronize();

  alloc_spin<<<1, 1, 1, stream1>>>(shared_ptr);
  cudaStream_t stream2;
  cudaStreamCreate(&stream2);

  read_spin<<<1, 1, 1, stream2>>>(shared_ptr);

  cudaStreamSynchronize(stream1);
  cudaStreamSynchronize(stream2);

  gpuErrchk(cudaGetLastError());

  return 0;
}