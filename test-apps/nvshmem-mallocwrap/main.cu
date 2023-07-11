#include <mpi.h>
#include <nvshmem.h>
#include <nvshmemx.h>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <sstream>
#include <cstdlib>
#include "cuda_wrapper.hpp"

#define SIZE 1

#define CUDA_CHECK(stmt)                                  \
do {                                                      \
    cudaError_t result = (stmt);                          \
    if (cudaSuccess != result) {                          \
        fprintf(stderr, "[%s:%d] CUDA failed with %s \n", \
         __FILE__, __LINE__, cudaGetErrorString(result)); \
        exit(-1);                                         \
    }                                                     \
} while (0)

__global__ void simple_shift(int *destination) {
    int mype = nvshmem_my_pe();
    int npes = nvshmem_n_pes();
    int peer = (mype + 1) % npes;
    

    for (int i = 0; i < SIZE; i++) {
        nvshmem_int_p(destination + i * sizeof(int), mype, peer);
    }
}

int main (int argc, char *argv[]) {
    int mype_node, msg;
    cudaStream_t stream;
    MPI_Comm mpi_comm = MPI_COMM_WORLD;;

    // before
    int rank = 0, nranks = 1;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

    int num_devices;
    cudaGetDeviceCount(&num_devices);

    int local_rank = -1;
    int local_size = 1;
    {
        MPI_Comm local_comm;
        MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL,
                                     &local_comm);

        MPI_Comm_rank(local_comm, &local_rank);
        MPI_Comm_size(local_comm, &local_size);

        MPI_Comm_free(&local_comm);
    }
    if ( 1 < num_devices && num_devices < local_size )
    {
        fprintf(stderr,"ERROR Number of visible devices (%d) is less than number of ranks on the node (%d)!\n", num_devices, local_size);
        MPI_Finalize();
        return 1;
    }
    if ( 1 == num_devices ) {
        // Only 1 device visible, assuming GPU affinity is handled via CUDA_VISIBLE_DEVICES
        cudaSetDevice(0);
    } else {
        cudaSetDevice(local_rank);
    }
    cudaFree(0);

    nvshmemx_init_attr_t attr;

    attr.mpi_comm = &mpi_comm;
    // after
    nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr);
    mype_node = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);

    CUDA_CHECK(cudaSetDevice(rank));
    CUDA_CHECK(cudaStreamCreate(&stream));
    int *destination = (int *) nvshmem_mallocWRAP (SIZE * sizeof(int), "destination", 4);
    int *aligned_var = (int *) nvshmem_alignWRAP (64, SIZE * sizeof(int), "aligned_var", 4);

    simple_shift<<<1, 1, 0, stream>>>(destination);
    nvshmemx_barrier_all_on_stream(stream);
    CUDA_CHECK(cudaMemcpyAsync(&msg, destination, sizeof(int),
                cudaMemcpyDeviceToHost, stream));

    CUDA_CHECK(cudaStreamSynchronize(stream));
    printf("%d: received message %d\n", nvshmem_my_pe(), msg);

    nvshmem_free(destination);
    nvshmem_free(aligned_var);
    nvshmem_finalize();
    MPI_Finalize();
    return 0;
}
