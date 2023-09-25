#include <getopt.h>
#include <iostream>
#include <stdio.h>
#include <unistd.h>

using namespace std;

#define gpuErrchk(ans)                                                         \
  { gpuAssert(ans); }
inline void gpuAssert(cudaError_t code) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s\n", cudaGetErrorString(code));
  }
}

__global__ void set_gpu_vals(int size, int *arr, int val) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;

  arr[idx] = val;
}

struct hipa_args {
  int size = 32;
  int verbose = 0;
  int check = 0;
  int async = 0;
};

typedef struct hipa_args hipa_args;

void getargs(hipa_args *args, int argc, char *argv[]) {
  int c;

  while ((c = getopt(argc, argv, "n:avc")) != -1) {
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
    case 'a':
      args->async = 1;
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

hipa_args *default_args() {
  hipa_args *args = (hipa_args *)malloc(sizeof(hipa_args));

  args->size = 32;
  args->verbose = 0;
  args->check = 0;
  args->async = 0;

  return args;
}

int main(int argc, char *argv[]) {

  hipa_args *args = default_args();
  getargs(args, argc, argv);

  int gpuid[] = {0, 1};

  cudaSetDevice(gpuid[0]);
  gpuErrchk(cudaDeviceEnablePeerAccess(1, 0));
  cudaSetDevice(gpuid[1]);
  gpuErrchk(cudaDeviceEnablePeerAccess(0, 0));
  cudaSetDevice(gpuid[0]);

  const size_t buf_size = args->size * sizeof(int);

  int *g0 = NULL;
  cudaSetDevice(gpuid[0]);
  gpuErrchk(cudaMalloc(&g0, buf_size));
  cudaSetDevice(gpuid[0]);

  int *g1 = NULL;
  cudaSetDevice(gpuid[1]);
  gpuErrchk(cudaMalloc(&g1, buf_size));
  cudaSetDevice(gpuid[0]);

  int *h0 = NULL;
  gpuErrchk(cudaMallocHost(&h0, buf_size));

  int *h1 = NULL;
  gpuErrchk(cudaMallocHost(&h1, buf_size));

  set_gpu_vals<<<std::ceil(args->size / 1024.0),
                 max(args->size > 1024 ? 1024 : args->size % 1025, 1)>>>(
      args->size, g0, 10);
  gpuErrchk(cudaDeviceSynchronize());

  if (args->async) {
    gpuErrchk(cudaMemcpyAsync(g1, g0, buf_size, cudaMemcpyDeviceToDevice));
  } else {
    gpuErrchk(cudaMemcpy(g1, g0, buf_size, cudaMemcpyDeviceToDevice));
  }

  if (args->check) {
    gpuErrchk(cudaMemcpy(h0, g0, buf_size, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(h1, g1, buf_size, cudaMemcpyDeviceToHost));

    for (int i = 0; i < args->size; i++) {
      if (args->verbose) {
        printf("\rchecking correctness against CPU: %.2f",
               ((float)(i + 1) / (float)args->size) * 100);

        if (i == args->size - 1) {
          printf("\n");
        }
      }

      if (h1[i] == h0[i]) {
        continue;
      }

      cout << "FAILED: (H0: " << i << ") " << h0[i] << "  != (H1: " << i << ") "
           << h1[i] << endl;
      return 1;
    }
  }

  cudaFree(h0);
  cudaFree(h1);
  cudaFree(g0);
  cudaFree(g1);

  free(args);

  return 0;
}
