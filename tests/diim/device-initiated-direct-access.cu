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

__global__ void simple_kernel(int size, int *src, int *dst1, int *dst2){
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size) {
    return;
  }

  if (idx % 2 == 0) {
    dst1[idx] = modify_cell(src[idx]);
  }
  else {
    dst2[idx] = modify_cell(src[idx]);
  }
}

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

  int gpuid[] = {0, 1, 2};

  int canAccessPeer;
  cudaError_t err = cudaDeviceCanAccessPeer(&canAccessPeer, 0, 1);
  gpuErrchk(cudaDeviceEnablePeerAccess(1, 0));
  gpuErrchk(cudaDeviceEnablePeerAccess(2, 0));
  gpuErrchk(cudaDeviceEnablePeerAccess(3, 0));

  const size_t buf_size = args->size * sizeof(int);

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
  simple_kernel<<<std::ceil(args->size / 1024.0), max(args->size > 1024 ? 1024 :args->size % 1025, 1)>>>(args->size, g0, g1, g2);
  cudaDeviceSynchronize();

  if (args->check) {
    gpuErrchk(cudaMemcpy(h1, g1, buf_size, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(h2, g2, buf_size, cudaMemcpyDeviceToHost));
    for (int i = 0; i < args->size; i++) {
      if (args->verbose) {
        printf("\rchecking correctness against CPU: %.2f", ((float) (i + 1) / (float) args->size) * 100);

        if (i == args->size - 1) {
          printf("\n");
        }
      }

      if (i % 2 == 0 && h1[i] == modify_cell(h0[i])) {
        continue;
      } else if (i % 2 == 1 && h2[i] == modify_cell(h0[i])) {
        continue;
      }

      cout << "FAILED: modify_cell((H0: " << i << ")) " << modify_cell(h0[i]) << "  != (H1: " << i << ") " << h1[i] << endl;
      return 1;
    }
  }


  cudaFree(h0);
  cudaFree(h1);
  cudaFree(h2);
  cudaFree(g0);
  cudaFree(g1);
  cudaFree(g2);

  free(args);

  return 0;
}
