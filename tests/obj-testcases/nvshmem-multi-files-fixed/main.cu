#include <stdio.h>
#include <getopt.h>
#include "mpi.h"
#include "nvshmem.h"
#include "nvshmemx.h"

#define CUDA_CHECK(stmt)                                  \
do {                                                      \
    cudaError_t result = (stmt);                          \
    if (cudaSuccess != result) {                          \
        fprintf(stderr, "[%s:%d] CUDA failed with %s \n", \
         __FILE__, __LINE__, cudaGetErrorString(result)); \
        exit(-1);                                         \
    }                                                     \
} while (0)

__global__ void simple_shift(int size, int *destination) {
    int mype = nvshmem_my_pe();
    int npes = nvshmem_n_pes();
    int peer = (mype + 1) % npes;
    

    int messages_per_node = size / npes;
    int extra_messages = size % npes;

    int n = 0;
    if (mype < extra_messages) {
      n = messages_per_node + 1;
    } else {
      n = messages_per_node;
    }

    for (int i = 0; i < n; i++) {
        nvshmem_int_p(destination + i * sizeof(int), mype, peer);
    }
}

struct nvshmem_args {
  int verbose = 0;
  int size = 32;
};

typedef struct nvshmem_args nvshmem_args;


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

int main (int argc, char *argv[]) {
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
    int *destination = (int *) nvshmem_malloc (args->size * sizeof(int));
    int *aligned_var = (int *) nvshmem_align  (64, args->size * sizeof(int));


    simple_shift<<<1, 1, 0, stream>>>(args->size, destination);
    nvshmemx_barrier_all_on_stream(stream);
    CUDA_CHECK(cudaMemcpyAsync(&msg, destination, sizeof(int),
                cudaMemcpyDeviceToHost, stream));

    CUDA_CHECK(cudaStreamSynchronize(stream));

    if (args->verbose) {
      printf("%d: received message %d\n", nvshmem_my_pe(), msg);
    }

    nvshmem_free(destination);
    nvshmem_free(aligned_var);
    nvshmem_finalize();
    MPI_Finalize();

    free(args);

    return 0;
}
