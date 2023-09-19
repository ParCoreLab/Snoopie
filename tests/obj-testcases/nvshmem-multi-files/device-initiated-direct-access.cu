#include <iostream>
#include <stdio.h>
#include <unistd.h>
#include <getopt.h>
#include "mpi.h"
#include "nvshmem.h"
#include "nvshmemx.h"

using namespace std;

struct nvshmem_args {
  int verbose = 0;
  int size = 32;
};

typedef struct nvshmem_args nvshmem_args;

#define CUDA_CHECK(stmt)                                  \
do {                                                      \
    cudaError_t result = (stmt);                          \
    if (cudaSuccess != result) {                          \
        fprintf(stderr, "[%s:%d] CUDA failed with %s \n", \
         __FILE__, __LINE__, cudaGetErrorString(result)); \
        exit(-1);                                         \
    }                                                     \
} while (0)

#define gpuErrchk(ans) { gpuAssert(ans); }
inline void gpuAssert(cudaError_t code)
{
  if (code != cudaSuccess) {
  fprintf(stderr,"GPUassert: %s\n", cudaGetErrorString(code));
  }
}

void getargs(nvshmem_args *args, int argc, char* argv[]) {
  int c;

  while ((c = getopt(argc, argv, "n:v")) != -1) {
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

nvshmem_args *default_args() {
  nvshmem_args *args = (nvshmem_args*) malloc(sizeof(nvshmem_args));

  args->verbose = 0;
  args->size = 32;

  return args;
}

__host__ __device__ int modify_cell(int a) {
  return a + 2;
}

__global__ void simple_kernel(int size, int *data/*, int *src, int *dst1, int *dst2, int *dst3*/){
    int mype = nvshmem_my_pe();
    int npes = nvshmem_n_pes();
    int peer = (mype + 1) % npes;

    nvshmem_int_p(data, mype, peer);
}

extern __global__ void simple_kernel1(int size, int *data);

extern __global__ void simple_kernel2(int size, int *data);

int main(int argc, char* argv[]) {

  nvshmem_args *args = default_args();
  getargs(args, argc, argv);

  int mype_node, msg;
  cudaStream_t stream;
    int rank, nranks;
    MPI_Comm mpi_comm = MPI_COMM_WORLD;
    nvshmemx_init_attr_t attr;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

    attr.mpi_comm = &mpi_comm;
    nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr);
    mype_node = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
    
    CUDA_CHECK(cudaSetDevice(mype_node));
    CUDA_CHECK(cudaStreamCreate(&stream));

  const size_t buf_size = args->size * sizeof(int);

  //int *g0 = NULL;
  //gpuErrchk(cudaMalloc(&g0, buf_size));
  int *g0 = (int *) nvshmem_malloc (buf_size * sizeof(int));

  //int *g1 = NULL;
  //gpuErrchk(cudaMalloc(&g1, buf_size));
  int *g1 = (int *) nvshmem_malloc (buf_size * sizeof(int));

  //int *g2 = NULL;
  //gpuErrchk(cudaMalloc(&g2, buf_size));
  int *g2 = (int *) nvshmem_malloc (buf_size * sizeof(int));

  //int *g3 = NULL;
  //gpuErrchk(cudaMalloc(&g3, buf_size));
  //int *g3 = (int *) nvshmem_malloc (buf_size * sizeof(int));

#if 0
  int *h0 = NULL;
  gpuErrchk(cudaMallocHost(&h0, buf_size));

  int *h1 = NULL;
  gpuErrchk(cudaMallocHost(&h1, buf_size));

  int *h2 = NULL;
  gpuErrchk(cudaMallocHost(&h2, buf_size));

  int *h3 = NULL; 
  gpuErrchk(cudaMallocHost(&h3, buf_size));


  gpuErrchk(cudaMemcpy(g0, h0, buf_size, cudaMemcpyHostToDevice));

  gpuErrchk(cudaMemcpy(g1, h1, buf_size, cudaMemcpyHostToDevice));

  gpuErrchk(cudaMemcpy(g2, h2, buf_size, cudaMemcpyHostToDevice));

  gpuErrchk(cudaMemcpy(g3, h3, buf_size, cudaMemcpyHostToDevice));
#endif
  simple_kernel<<<1, 1>>>(args->size, g0/*, g1, g2, g3*/);
  simple_kernel1<<<1, 1>>>(args->size, g1/*, g1, g2, g3*/);
  simple_kernel2<<<1, 1>>>(args->size, g2/*, g1, g2, g3*/);
  cudaDeviceSynchronize(); 

  
  //cudaFree(h0);
  //cudaFree(h1);
  //cudaFree(h2);
  //cudaFree(h3);
  nvshmem_free(g0);
  nvshmem_free(g1);
  nvshmem_free(g2);

  nvshmem_finalize();
  MPI_Finalize();

  free(args);

  return 0;
}
