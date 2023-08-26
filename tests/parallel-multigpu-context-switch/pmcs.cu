#include <iostream>
#include <omp.h>
#include <cuda.h>

#define CUDA_RT_CALL(call)                                                                  \
    {                                                                                       \
        cudaError_t cudaStatus = call;                                                      \
        if (cudaSuccess != cudaStatus)                                                      \
            fprintf(stderr,                                                                 \
                    "ERROR: CUDA RT call \"%s\" in line %d of file %s failed "              \
                    "with "                                                                 \
                    "%s (%d).\n",                                                           \
                    #call, __LINE__, __FILE__, cudaGetErrorString(cudaStatus), cudaStatus); \
    }                                                                                       \
    
int main() {
    std::cout << "Started multi context in parallel" << std::endl;

    int num_devices = 0;
    CUDA_RT_CALL(cudaGetDeviceCount(&num_devices));

#pragma omp parallel num_threads(num_devices)
    {
        int dev_id = omp_get_thread_num();
        std::cout << "Switching to device: " << dev_id << std::endl;
#pragma omp barrier
        CUDA_RT_CALL(cudaSetDevice(dev_id));
        // CUDA_RT_CALL(cudaFree(0));
        CUcontext ctx;
        cuDevicePrimaryCtxRetain(&ctx, dev_id);
#pragma omp barrier
        std::cout << "Switched to device: "  << dev_id << std::endl;
    }

    std::cout << "Ended multi context in parallel" << std::endl;

    return 0;
}
