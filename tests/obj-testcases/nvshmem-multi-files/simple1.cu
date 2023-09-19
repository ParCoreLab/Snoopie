#include "nvshmem.h"
#include "nvshmemx.h"

extern __host__ __device__ int modify_cell(int a);

__global__ void simple_kernel1(int size, int *data){
    int mype = nvshmem_my_pe();
    int npes = nvshmem_n_pes();
    int peer = ((mype - 1) >= 0) ? (mype - 1)  : (npes - 1);

    nvshmem_int_p(data, mype, peer); 
}

