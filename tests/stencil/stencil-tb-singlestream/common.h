#ifndef INC_2D_STENCIL_COMMON_H
#define INC_2D_STENCIL_COMMON_H

#include <omp.h>

#include <algorithm>
#include <array>
#include <sstream>
#include <string>

#include <cooperative_groups.h>

#include <cmath>
#include <cstdio>
#include <iostream>

typedef float real;

typedef int (*initfunc_t)(int argc, char **argv);

constexpr int MAX_NUM_DEVICES{32};
constexpr real tol = 1.0e-7;
const real PI{static_cast<real>(2.0 * std::asin(1.0))};
constexpr int MAX_NUM_ELEM_PER_GPU = 256 * 256;
constexpr int TILE_SIZE = 256;

template <typename T>
T get_argval(char **begin, char **end, const std::string &arg, const T default_val) {
    T argval = default_val;
    char **itr = std::find(begin, end, arg);
    if (itr != end && ++itr != end) {
        std::istringstream inbuf(*itr);
        inbuf >> argval;
    }
    return argval;
}

bool get_arg(char **begin, char **end, const std::string &arg);

__global__ void initialize_boundaries(real *__restrict__ const a_new, real *__restrict__ const a,
                                      const real pi, const int offset, const int nx,
                                      const int my_ny, const int ny);

__global__ void jacobi_kernel_single_gpu(real *__restrict__ const a_new,
                                         const real *__restrict__ const a,
                                         real *__restrict__ const l2_norm, const int iy_start,
                                         const int iy_end, const int nx, const bool calculate_norm);

__global__ void jacobi_kernel_single_gpu_mirror(real *__restrict__ const a_new,
                                                const real *__restrict__ const a,
                                                real *__restrict__ const l2_norm,
                                                const int iy_start, const int iy_end, const int nx,
                                                const bool calculate_norm);

__global__ void jacobi_kernel_single_gpu_perks(real *__restrict__ const a_new,
                                               const real *__restrict__ const a,
                                               real *__restrict__ const l2_norm, const int iy_start,
                                               const int iy_end, const int nx,
                                               const bool calculate_norm);

double single_cpu(real *a_h_input, const int nx, const int ny, const int iter_max,
                  real *const a_ref_h, const int nccheck, const bool print);

double single_gpu(const int nx, const int ny, const int iter_max, real *const a_ref_h,
                  const int nccheck, const bool print,
                  decltype(jacobi_kernel_single_gpu) kernel = jacobi_kernel_single_gpu);

double single_gpu(real *input, const int nx, const int ny, const int iter_max, real *const a_ref_h,
                  const int nccheck, const bool print);

double single_gpu_persistent(const int nx, const int ny, const int iter_max, real *const a_ref_h,
                             const int nccheck, const bool print);

void report_results(const int ny, const int nx, real *a_ref_h, real *a_h, const int num_devices,
                    const double runtime_serial_non_persistent, const double start,
                    const double stop, const bool compare_to_single_gpu);

#define noop

#ifdef USE_NVTX
#include <nvToolsExt.h>

const uint32_t colors[] = {0x0000ff00, 0x000000ff, 0x00ffff00, 0x00ff00ff,
                           0x0000ffff, 0x00ff0000, 0x00ffffff};
const int num_colors = sizeof(colors) / sizeof(uint32_t);

#define PUSH_RANGE(name, cid)                              \
    {                                                      \
        int color_id = cid;                                \
        color_id = color_id % num_colors;                  \
        nvtxEventAttributes_t eventAttrib = {0};           \
        eventAttrib.version = NVTX_VERSION;                \
        eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;  \
        eventAttrib.colorType = NVTX_COLOR_ARGB;           \
        eventAttrib.color = colors[color_id];              \
        eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII; \
        eventAttrib.message.ascii = name;                  \
        nvtxRangePushEx(&eventAttrib);                     \
    }
#define POP_RANGE nvtxRangePop();
#else
#define PUSH_RANGE(name, cid)
#define POP_RANGE
#endif

#define MPI_CALL(call)                                                                \
    {                                                                                 \
        int mpi_status = call;                                                        \
        if (MPI_SUCCESS != mpi_status) {                                              \
            char mpi_error_string[MPI_MAX_ERROR_STRING];                              \
            int mpi_error_string_length = 0;                                          \
            MPI_Error_string(mpi_status, mpi_error_string, &mpi_error_string_length); \
            if (NULL != mpi_error_string)                                             \
                fprintf(stderr,                                                       \
                        "ERROR: MPI call \"%s\" in line %d of file %s failed "        \
                        "with %s "                                                    \
                        "(%d).\n",                                                    \
                        #call, __LINE__, __FILE__, mpi_error_string, mpi_status);     \
            else                                                                      \
                fprintf(stderr,                                                       \
                        "ERROR: MPI call \"%s\" in line %d of file %s failed "        \
                        "with %d.\n",                                                 \
                        #call, __LINE__, __FILE__, mpi_status);                       \
            exit(mpi_status);                                                         \
        }                                                                             \
    }

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
    static_assert(true, "")

#endif  // INC_2D_STENCIL_COMMON_H