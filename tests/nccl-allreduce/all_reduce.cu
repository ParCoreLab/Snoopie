#include "cuda_runtime.h"
#include "nccl.h"
#include <chrono>
#include <getopt.h>
#include <iostream>
#include <unistd.h>

#define CUDACHECK(cmd)                                                         \
  do {                                                                         \
    cudaError_t err = cmd;                                                     \
    if (err != cudaSuccess) {                                                  \
      printf("Failed: Cuda error %s:%d '%s'\n", __FILE__, __LINE__,            \
             cudaGetErrorString(err));                                         \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

#define NCCLCHECK(cmd)                                                         \
  do {                                                                         \
    ncclResult_t res = cmd;                                                    \
    if (res != ncclSuccess) {                                                  \
      printf("Failed, NCCL error %s:%d '%s'\n", __FILE__, __LINE__,            \
             ncclGetErrorString(res));                                         \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

struct all_reduce_args {
  int size = 32;
  int ngpus = 4;
  int check = 0;
};

typedef struct all_reduce_args all_reduce_args;

void getargs(all_reduce_args *args, int argc, char *argv[]) {
  int c;

  while ((c = getopt(argc, argv, "n:g:c")) != -1) {
    switch (c) {
    case 'n':
      args->size = atoi(optarg);
      if (args->size <= 0) {
        fprintf(stderr, "Error: argument for -n cannot be 0 or less\n");
      }
      break;
    case 'g':
      args->ngpus = atoi(optarg);
      if (args->ngpus <= 1) {
        fprintf(stderr, "Error: argument for -n cannot be 1 or less\n");
      }
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

all_reduce_args *default_args() {
  all_reduce_args *args = (all_reduce_args *)malloc(sizeof(all_reduce_args));

  args->size = 32;
  args->ngpus = 4;
  args->check = 0;

  return args;
}

int main(int argc, char *argv[]) {
  all_reduce_args *args = default_args();
  getargs(args, argc, argv);

  ncclComm_t comms[4];

  // managing 4 devices
  int nDev = args->ngpus;
  int size = args->size;
  int *devs = (int *)malloc(sizeof(int) * args->ngpus);
  for (int i = 0; i < args->ngpus; i++) {
    devs[i] = i;
  }

  std::cout << "Size: " << size << ", NGPUS: " << nDev << std::endl;

  // allocating and initializing device buffers
  float **sendbuff = (float **)malloc(nDev * sizeof(float *));
  float **recvbuff = (float **)malloc(nDev * sizeof(float *));
  cudaStream_t *s = (cudaStream_t *)malloc(sizeof(cudaStream_t) * nDev);

  for (int i = 0; i < nDev; ++i) {
    CUDACHECK(cudaSetDevice(i));
    CUDACHECK(cudaMalloc(sendbuff + i, size * sizeof(float)));
    CUDACHECK(cudaMalloc(recvbuff + i, size * sizeof(float)));
    CUDACHECK(cudaMemset(sendbuff[i], 1, size * sizeof(float)));
    CUDACHECK(cudaMemset(recvbuff[i], 0, size * sizeof(float)));
    CUDACHECK(cudaStreamCreate(s + i));
  }

  // initializing NCCL
  NCCLCHECK(ncclCommInitAll(comms, nDev, devs));

  // calling NCCL communication API. Group API is required when using
  // multiple devices per thread
  NCCLCHECK(ncclGroupStart());

  auto start = std::chrono::high_resolution_clock::now();

  for (int i = 0; i < nDev; ++i)
    NCCLCHECK(ncclAllReduce((const void *)sendbuff[i], (void *)recvbuff[i],
                            size, ncclFloat, ncclSum, comms[i], s[i]));

  NCCLCHECK(ncclGroupEnd());

  for (int i = 0; i < nDev; ++i) {
    CUDACHECK(cudaSetDevice(i));
    CUDACHECK(cudaStreamSynchronize(s[i]));
  }

  auto end = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start)
          .count();

  std::cout << "Time taken: " << duration << " microseconds." << std::endl;

  // synchronizing on CUDA streams to wait for completion of NCCL operation
  for (int i = 0; i < nDev; ++i) {
    CUDACHECK(cudaSetDevice(i));
    CUDACHECK(cudaStreamSynchronize(s[i]));
  }

  // free device buffers
  for (int i = 0; i < nDev; ++i) {
    CUDACHECK(cudaSetDevice(i));
    CUDACHECK(cudaFree(sendbuff[i]));
    CUDACHECK(cudaFree(recvbuff[i]));
  }

  // finalizing NCCL
  for (int i = 0; i < nDev; ++i)
    ncclCommDestroy(comms[i]);

  free(devs);

  printf("Success \n");
  return 0;
}
