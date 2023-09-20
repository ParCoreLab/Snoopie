#include <iostream>
#include <stdio.h>
#include <unistd.h>
#include <getopt.h>

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

__global__ void simple_kernel(int size, int *src, int *dst1, int *dst2, int *dst3){
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size) { 
    return;
  }

  if (idx % 3 == 0) {
    dst1[idx] = modify_cell(src[idx]);
  }
  else if (idx % 3 == 1) {
    dst2[idx] = modify_cell(src[idx]);
  } else {
    dst3[idx] = modify_cell(src[idx]);
  }
}

extern __global__ void simple_kernel1(int size, int *src, int *dst1, int *dst2, int *dst3);

extern __global__ void simple_kernel2(int size, int *src, int *dst1, int *dst2, int *dst3);

struct diim_args {
  int size = 32;
  int verbose = 0;
  int check = 0;
};

typedef struct diim_args diim_args;


void getargs(diim_args *args, int argc, char* argv[]) {
  int c;

  while ((c = getopt(argc, argv, "n:v:c")) != -1) {
    switch (c) {
      case 'n':
        args->size = atoi(optarg);
        if (args->size <= 0) {
          fprintf(stderr, "Error: argument for -n cannot be 0 or less\n");
        }
        break;
      case 'v':
        args->verbose = 1;
        break;
      case 'c':
        args->check = 1;
        break;
      case '?':
        if (optopt == 'n') {
          fprintf(stderr, "Error: no argument provided for -n flag\n");
        } else {
          fprintf(stderr, "Error: unknown option '%c'\n", optopt);
        }
        exit(1);
      default:
        abort();
    }
  }
}

diim_args *default_args() {
  diim_args *args = (diim_args*) malloc(sizeof(diim_args));

  args->size = 32;
  args->verbose = 0;
  args->check = 0;
  
  return args;
}

int main(int argc, char* argv[]) {

  diim_args *args = default_args();
  getargs(args, argc, argv);

  int gpuid[] = {0, 1, 2, 3};

  cudaSetDevice(gpuid[0]);
  gpuErrchk(cudaDeviceEnablePeerAccess(1, 0));
  gpuErrchk(cudaDeviceEnablePeerAccess(2, 0));
  gpuErrchk(cudaDeviceEnablePeerAccess(3, 0));

  cudaSetDevice(gpuid[1]);
  gpuErrchk(cudaDeviceEnablePeerAccess(0, 0));
  gpuErrchk(cudaDeviceEnablePeerAccess(2, 0));
  gpuErrchk(cudaDeviceEnablePeerAccess(3, 0));

  cudaSetDevice(gpuid[2]);
  gpuErrchk(cudaDeviceEnablePeerAccess(0, 0));
  gpuErrchk(cudaDeviceEnablePeerAccess(1, 0));
  gpuErrchk(cudaDeviceEnablePeerAccess(3, 0));

  cudaSetDevice(gpuid[3]);
  gpuErrchk(cudaDeviceEnablePeerAccess(0, 0));
  gpuErrchk(cudaDeviceEnablePeerAccess(1, 0));
  gpuErrchk(cudaDeviceEnablePeerAccess(2, 0));

  const size_t buf_size = args->size * sizeof(int);

  int *g0 = NULL;
  cudaSetDevice(gpuid[0]);
  gpuErrchk(cudaMalloc(&g0, buf_size));

  int *g1 = NULL;

  cudaSetDevice(gpuid[1]);
  gpuErrchk(cudaMalloc(&g1, buf_size));

  int *g2 = NULL;

  cudaSetDevice(gpuid[2]);
  gpuErrchk(cudaMalloc(&g2, buf_size));

  int *g3 = NULL;
  
  cudaSetDevice(gpuid[3]);
  gpuErrchk(cudaMalloc(&g3, buf_size));

  cudaSetDevice(gpuid[0]);
  int *h0 = NULL;
  gpuErrchk(cudaMallocHost(&h0, buf_size));

  cudaSetDevice(gpuid[1]);
  int *h1 = NULL;
  gpuErrchk(cudaMallocHost(&h1, buf_size));

  cudaSetDevice(gpuid[2]);
  int *h2 = NULL;
  gpuErrchk(cudaMallocHost(&h2, buf_size));

  cudaSetDevice(gpuid[3]);
  int *h3 = NULL; 
  gpuErrchk(cudaMallocHost(&h3, buf_size));


  cudaSetDevice(gpuid[0]);
  gpuErrchk(cudaMemcpy(g0, h0, buf_size, cudaMemcpyHostToDevice));

  cudaSetDevice(gpuid[1]);
  gpuErrchk(cudaMemcpy(g1, h1, buf_size, cudaMemcpyHostToDevice));

  cudaSetDevice(gpuid[2]);
  gpuErrchk(cudaMemcpy(g2, h2, buf_size, cudaMemcpyHostToDevice));

  cudaSetDevice(gpuid[3]);
  gpuErrchk(cudaMemcpy(g3, h3, buf_size, cudaMemcpyHostToDevice));

  cudaSetDevice(gpuid[0]);
  simple_kernel<<<std::ceil(args->size / 1024.0), max(args->size > 1024 ? 1024 :args->size % 1025, 1)>>>(args->size, g0, g1, g2, g3);
  simple_kernel1<<<std::ceil(args->size / 1024.0), max(args->size > 1024 ? 1024 :args->size % 1025, 1)>>>(args->size, g0, g1, g2, g3);
  simple_kernel2<<<std::ceil(args->size / 1024.0), max(args->size > 1024 ? 1024 :args->size % 1025, 1)>>>(args->size, g0, g1, g2, g3);
  cudaDeviceSynchronize();

  if (args->check) {
    gpuErrchk(cudaMemcpy(h1, g1, buf_size, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(h2, g2, buf_size, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(h3, g2, buf_size, cudaMemcpyDeviceToHost));
  }

  
  cudaFree(h0);
  cudaFree(h1);
  cudaFree(h2);
  cudaFree(h3);
  cudaFree(g0);
  cudaFree(g1);
  cudaFree(g2);
  cudaFree(g3);
  free(args);

  return 0;
}
