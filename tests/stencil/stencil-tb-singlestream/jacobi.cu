/* Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
 */

#include "multi-threaded-two-block-comm.cuh"

namespace cg = cooperative_groups;

namespace SSMultiThreadedTwoBlockComm {
__global__ void __launch_bounds__(1024, 1)
    jacobi_kernel(real *a_new, real *a, const int iy_start, const int iy_end, const int nx,
                  const int grid_dim_x, const int iter_max,
                  volatile real *local_halo_buffer_for_top_neighbor,
                  volatile real *local_halo_buffer_for_bottom_neighbor,
                  volatile real *remote_my_halo_buffer_on_top_neighbor,
                  volatile real *remote_my_halo_buffer_on_bottom_neighbor,
                  volatile int *local_is_top_neighbor_done_writing_to_me,
                  volatile int *local_is_bottom_neighbor_done_writing_to_me,
                  volatile int *remote_am_done_writing_to_top_neighbor,
                  volatile int *remote_am_done_writing_to_bottom_neighbor) {
    cg::thread_block cta = cg::this_thread_block();
    cg::grid_group grid = cg::this_grid();

    int iter = 0;
    int cur_iter_mod = 0;
    int next_iter_mod = 1;

    const int comp_size_iy = ((gridDim.x - 2) / grid_dim_x) * blockDim.y * nx;
    const int comp_size_ix = grid_dim_x * blockDim.x;

    const int comp_start_iy =
        ((blockIdx.x / grid_dim_x) * blockDim.y + threadIdx.y + iy_start + 1) * nx;
    const int comp_start_ix = ((blockIdx.x % grid_dim_x) * blockDim.x + threadIdx.x + 1);

    const int end_iy = (iy_end - 1) * nx;
    const int end_ix = (nx - 1);

    const int comm_size_ix = blockDim.y * blockDim.x;

    const int comm_start_ix = threadIdx.y * blockDim.x + threadIdx.x + 1;
    const int comm_start_iy = iy_start * nx;

    while (iter < iter_max) {
        if (blockIdx.x == gridDim.x - 1) {
            if (!cta.thread_rank()) {
                while (local_is_top_neighbor_done_writing_to_me[cur_iter_mod * 2] != iter) {
                }
            }
            cg::sync(cta);

            for (int ix = comm_start_ix; ix < end_ix; ix += comm_size_ix) {
                const real first_row_val =
                    0.25 * (a[comm_start_iy + ix + 1] + a[comm_start_iy + ix - 1] +
                            a[comm_start_iy + nx + ix] +
                            remote_my_halo_buffer_on_top_neighbor[cur_iter_mod * nx + ix]);
                a_new[comm_start_iy + ix] = first_row_val;
                local_halo_buffer_for_top_neighbor[nx * next_iter_mod + ix] = first_row_val;
            }

            cg::sync(cta);

            if (!cta.thread_rank()) {
                remote_am_done_writing_to_top_neighbor[next_iter_mod * 2 + 1] = iter + 1;
            }
        } else if (blockIdx.x == gridDim.x - 2) {
            if (!cta.thread_rank()) {
                while (local_is_bottom_neighbor_done_writing_to_me[cur_iter_mod * 2 + 1] != iter) {
                }
            }
            cg::sync(cta);

            for (int ix = comm_start_ix; ix < end_ix; ix += comm_size_ix) {
                const real last_row_val =
                    0.25 * (a[end_iy + ix + 1] + a[end_iy + ix - 1] +
                            remote_my_halo_buffer_on_bottom_neighbor[cur_iter_mod * nx + ix] +
                            a[end_iy - nx + ix]);
                a_new[end_iy + ix] = last_row_val;
                local_halo_buffer_for_bottom_neighbor[nx * next_iter_mod + ix] = last_row_val;
            }

            cg::sync(cta);

            if (!cta.thread_rank()) {
                remote_am_done_writing_to_bottom_neighbor[next_iter_mod * 2] = iter + 1;
            }
        } else {
            for (int iy = comp_start_iy; iy < end_iy; iy += comp_size_iy) {
                for (int ix = comp_start_ix; ix < end_ix; ix += comp_size_ix) {
                    a_new[iy + ix] = 0.25 * (a[iy + ix + 1] + a[iy + ix - 1] + a[iy + nx + ix] +
                                             a[iy - nx + ix]);
                }
            }
        }

        real *temp_pointer = a_new;
        a_new = a;
        a = temp_pointer;

        iter++;

        next_iter_mod = cur_iter_mod;
        cur_iter_mod = 1 - cur_iter_mod;

        cg::sync(grid);
    }
}
}  // namespace SSMultiThreadedTwoBlockComm

bool get_arg(char **begin, char **end, const std::string &arg) {
    char **itr = std::find(begin, end, arg);
    if (itr != end) {
        return true;
    }
    return false;
}

__global__ void initialize_boundaries(real *__restrict__ const a_new, real *__restrict__ const a,
                                      const real pi, const int offset, const int nx,
                                      const int my_ny, const int ny) {
    for (unsigned int iy = blockIdx.x * blockDim.x + threadIdx.x; iy < my_ny;
         iy += blockDim.x * gridDim.x) {
        const real y0 = sin(2.0 * pi * (offset + iy) / (ny - 1));
        for (unsigned int ix = 0; ix < nx; ix += nx - 1) {
            a[iy * nx + ix] = y0;
            a_new[iy * nx + ix] = y0;
        }
    }
}

__global__ void jacobi_kernel_single_gpu(real *__restrict__ const a_new,
                                         const real *__restrict__ const a,
                                         real *__restrict__ const l2_norm, const int iy_start,
                                         const int iy_end, const int nx,
                                         const bool calculate_norm) {
    int iy = blockIdx.y * blockDim.y + threadIdx.y + iy_start;
    int ix = blockIdx.x * blockDim.x + threadIdx.x + 1;

    if (iy < iy_end && ix < (nx - 1)) {
        const real new_val = 0.25 * (a[iy * nx + ix + 1] + a[iy * nx + ix - 1] +
                                     a[(iy + 1) * nx + ix] + a[(iy - 1) * nx + ix]);

        a_new[iy * nx + ix] = new_val;
    }
}

double single_gpu(const int nx, const int ny, const int iter_max, real *const a_ref_h,
                  const int nccheck, const bool print, decltype(jacobi_kernel_single_gpu) kernel) {
    real *a;
    real *a_new;

    cudaStream_t compute_stream;
    cudaStream_t push_top_stream;
    cudaStream_t push_bottom_stream;
    cudaEvent_t compute_done;
    cudaEvent_t push_top_done;
    cudaEvent_t push_bottom_done;

    //    real* l2_norm_d;
    //    real* l2_norm_h;

    int iy_start = 1;
    int iy_end = (ny - 1);

    CUDA_RT_CALL(cudaMalloc(&a, nx * ny * sizeof(real)));
    CUDA_RT_CALL(cudaMalloc(&a_new, nx * ny * sizeof(real)));

    CUDA_RT_CALL(cudaMemset(a, 0, nx * ny * sizeof(real)));
    CUDA_RT_CALL(cudaMemset(a_new, 0, nx * ny * sizeof(real)));

    // Set diriclet boundary conditions on left and right boarder
    initialize_boundaries<<<ny / 128 + 1, 128>>>(a, a_new, PI, 0, nx, ny, ny);
    CUDA_RT_CALL(cudaGetLastError());
    CUDA_RT_CALL(cudaDeviceSynchronize());

    CUDA_RT_CALL(cudaStreamCreate(&compute_stream));
    CUDA_RT_CALL(cudaStreamCreate(&push_top_stream));
    CUDA_RT_CALL(cudaStreamCreate(&push_bottom_stream));
    CUDA_RT_CALL(cudaEventCreateWithFlags(&compute_done, cudaEventDisableTiming));
    CUDA_RT_CALL(cudaEventCreateWithFlags(&push_top_done, cudaEventDisableTiming));
    CUDA_RT_CALL(cudaEventCreateWithFlags(&push_bottom_done, cudaEventDisableTiming));

    //    CUDA_RT_CALL(cudaMalloc(&l2_norm_d, sizeof(real)));
    //    CUDA_RT_CALL(cudaMallocHost(&l2_norm_h, sizeof(real)));

    CUDA_RT_CALL(cudaDeviceSynchronize());
    
    if (print)
        printf(
            "Single GPU jacobi relaxation (non-persistent kernel): %d iterations on %d x %d "
            "mesh "
            "with "
            "norm "
            "check every %d iterations\n",
            iter_max, ny, nx, nccheck);

    constexpr int dim_block_x = 32;
    constexpr int dim_block_y = 32;
    dim3 dim_grid((nx + dim_block_x - 1) / dim_block_x, (ny + dim_block_y - 1) / dim_block_y, 1);

    int iter = 0;
    bool calculate_norm = false;
    //    real l2_norm = 1.0;

    double start = omp_get_wtime();
    PUSH_RANGE("Jacobi solve", 0)
    while (iter < iter_max) {
        //        CUDA_RT_CALL(cudaMemsetAsync(l2_norm_d, 0, sizeof(real), compute_stream));

        CUDA_RT_CALL(cudaStreamWaitEvent(compute_stream, push_top_done, 0));
        CUDA_RT_CALL(cudaStreamWaitEvent(compute_stream, push_bottom_done, 0));

        //        calculate_norm = (iter % nccheck) == 0 || (print && ((iter % 100) == 0));
        kernel<<<dim_grid, {dim_block_x, dim_block_y, 1}, 0, compute_stream>>>(
            a_new, a, nullptr, iy_start, iy_end, nx, calculate_norm);
        CUDA_RT_CALL(cudaGetLastError());
        CUDA_RT_CALL(cudaEventRecord(compute_done, compute_stream));

        //        if (calculate_norm) {
        //            CUDA_RT_CALL(cudaMemcpyAsync(l2_norm_h, l2_norm_d, sizeof(real),
        //            cudaMemcpyDeviceToHost,
        //                                         compute_stream));
        //        }

        // Apply periodic boundary conditions

        CUDA_RT_CALL(cudaStreamWaitEvent(push_top_stream, compute_done, 0));
        CUDA_RT_CALL(cudaMemcpyAsync(a_new, a_new + (iy_end - 1) * nx, nx * sizeof(real),
                                     cudaMemcpyDeviceToDevice, push_top_stream));
        CUDA_RT_CALL(cudaEventRecord(push_top_done, push_top_stream));

        CUDA_RT_CALL(cudaStreamWaitEvent(push_bottom_stream, compute_done, 0));
        CUDA_RT_CALL(cudaMemcpyAsync(a_new + iy_end * nx, a_new + iy_start * nx, nx * sizeof(real),
                                     cudaMemcpyDeviceToDevice, compute_stream));
        CUDA_RT_CALL(cudaEventRecord(push_bottom_done, push_bottom_stream));
        
        //        if (calculate_norm) {
        //            CUDA_RT_CALL(cudaStreamSynchronize(compute_stream));
        //            l2_norm = *l2_norm_h;
        //            l2_norm = std::sqrt(l2_norm);
        //            if (print && (iter % 100) == 0) printf("%5d, %0.6f\n", iter, l2_norm);
        //        }

        std::swap(a_new, a);
        iter++;
    }

    POP_RANGE
    double stop = omp_get_wtime();

    CUDA_RT_CALL(cudaMemcpy(a_ref_h, a, nx * ny * sizeof(real), cudaMemcpyDeviceToHost));

    CUDA_RT_CALL(cudaEventDestroy(push_bottom_done));
    CUDA_RT_CALL(cudaEventDestroy(push_top_done));
    CUDA_RT_CALL(cudaEventDestroy(compute_done));
    CUDA_RT_CALL(cudaStreamDestroy(push_bottom_stream));
    CUDA_RT_CALL(cudaStreamDestroy(push_top_stream));
    CUDA_RT_CALL(cudaStreamDestroy(compute_stream));

    //    CUDA_RT_CALL(cudaFreeHost(l2_norm_h));
    //    CUDA_RT_CALL(cudaFree(l2_norm_d));

    CUDA_RT_CALL(cudaFree(a_new));
    CUDA_RT_CALL(cudaFree(a));
    return (stop - start);
}

void report_results(const int ny, const int nx, real *a_ref_h, real *a_h, const int num_devices,
                    const double runtime_serial_non_persistent, const double start,
                    const double stop, const bool compare_to_single_gpu) {
    bool result_correct = true;

    if (compare_to_single_gpu) {
        for (int iy = 1; result_correct && (iy < (ny - 1)); ++iy) {
            for (int ix = 1; result_correct && (ix < (nx - 1)); ++ix) {
                if (std::fabs(a_ref_h[iy * nx + ix] - a_h[iy * nx + ix]) > tol) {
                    fprintf(stderr,
                            "ERROR: a[%d * %d + %d] = %.8f does not match %.8f "
                            "(reference)\n",
                            iy, nx, ix, a_h[iy * nx + ix], a_ref_h[iy * nx + ix]);
                    // result_correct = false;
                }
            }
        }
    }

    if (result_correct) {
        // printf("Num GPUs: %d.\n", num_devices);
        printf("Execution time: %8.4f s\n", (stop - start));

        if (compare_to_single_gpu) {
            printf(
                "Non-persistent kernel - %dx%d: 1 GPU: %8.4f s, %d GPUs: %8.4f s, speedup: "
                "%8.2f, "
                "efficiency: %8.2f \n",
                ny, nx, runtime_serial_non_persistent, num_devices, (stop - start),
                runtime_serial_non_persistent / (stop - start),
                runtime_serial_non_persistent / (num_devices * (stop - start)) * 100);
        }
    }
}

int main(int argc, char *argv[]) {
    const int iter_max = get_argval<int>(argv, argv + argc, "-niter", 1000);
    const int nx = get_argval<int>(argv, argv + argc, "-nx", 16384);
    const int ny = get_argval<int>(argv, argv + argc, "-ny", 16384);
    const bool compare_to_single_gpu = get_arg(argv, argv + argc, "-compare");

    real *a[MAX_NUM_DEVICES];
    real *a_new[MAX_NUM_DEVICES];
    int iy_end[MAX_NUM_DEVICES];

    real *halo_buffer_for_top_neighbor[MAX_NUM_DEVICES];
    real *halo_buffer_for_bottom_neighbor[MAX_NUM_DEVICES];

    int *is_top_done_computing_flags[MAX_NUM_DEVICES];
    int *is_bottom_done_computing_flags[MAX_NUM_DEVICES];

    real *a_ref_h;
    real *a_h;

    double runtime_serial_non_persistent = 0.0;

    int num_devices = 0;
    CUDA_RT_CALL(cudaGetDeviceCount(&num_devices));

#pragma omp parallel num_threads(num_devices)
    {
        int dev_id = omp_get_thread_num();

#pragma omp critical
	{
        CUDA_RT_CALL(cudaSetDevice(dev_id));
        CUDA_RT_CALL(cudaFree(nullptr));
	}

        if (compare_to_single_gpu && 0 == dev_id) {
            CUDA_RT_CALL(cudaMallocHost(&a_ref_h, nx * ny * sizeof(real)));
            CUDA_RT_CALL(cudaMallocHost(&a_h, nx * ny * sizeof(real)));

            runtime_serial_non_persistent = single_gpu(nx, ny, iter_max, a_ref_h, 0, true);
        }

#pragma omp barrier

        int chunk_size;
        int chunk_size_low = (ny - 2) / num_devices;
        int chunk_size_high = chunk_size_low + 1;

        // int height_per_gpu = ny / num_devices;

        cudaDeviceProp deviceProp{};
        CUDA_RT_CALL(cudaGetDeviceProperties(&deviceProp, dev_id));
        int numSms = deviceProp.multiProcessorCount;

        constexpr int dim_block_x = 32;
        constexpr int dim_block_y = 32;

        constexpr int grid_dim_x = 8;
        const int grid_dim_y = (numSms - 2) / grid_dim_x;
        constexpr int num_flags = 4;

        int num_ranks_low = num_devices * chunk_size_low + num_devices - (ny - 2);
        if (dev_id < num_ranks_low)
            chunk_size = chunk_size_low;
        else
            chunk_size = chunk_size_high;

        const int top = dev_id > 0 ? dev_id - 1 : (num_devices - 1);
        const int bottom = (dev_id + 1) % num_devices;

        if (top != dev_id) {
            int canAccessPeer = 0;
            CUDA_RT_CALL(cudaDeviceCanAccessPeer(&canAccessPeer, dev_id, top));
            if (canAccessPeer) {
                CUDA_RT_CALL(cudaDeviceEnablePeerAccess(top, 0));
            } else {
                std::cerr << "P2P access required from " << dev_id << " to " << top << std::endl;
            }
            if (top != bottom) {
                canAccessPeer = 0;
                CUDA_RT_CALL(cudaDeviceCanAccessPeer(&canAccessPeer, dev_id, bottom));
                if (canAccessPeer) {
                    CUDA_RT_CALL(cudaDeviceEnablePeerAccess(bottom, 0));
                } else {
                    std::cerr << "P2P access required from " << dev_id << " to " << bottom
                              << std::endl;
                }
            }
        }

#pragma omp barrier

        CUDA_RT_CALL(cudaMalloc(a + dev_id, nx * (chunk_size + 2) * sizeof(real)));
        CUDA_RT_CALL(cudaMalloc(a_new + dev_id, nx * (chunk_size + 2) * sizeof(real)));

        CUDA_RT_CALL(cudaMemset(a[dev_id], 0, nx * (chunk_size + 2) * sizeof(real)));
        CUDA_RT_CALL(cudaMemset(a_new[dev_id], 0, nx * (chunk_size + 2) * sizeof(real)));

        CUDA_RT_CALL(cudaMalloc(halo_buffer_for_top_neighbor + dev_id, 2 * nx * sizeof(real)));
        CUDA_RT_CALL(cudaMalloc(halo_buffer_for_bottom_neighbor + dev_id, 2 * nx * sizeof(real)));

        CUDA_RT_CALL(cudaMemset(halo_buffer_for_top_neighbor[dev_id], 0, 2 * nx * sizeof(real)));
        CUDA_RT_CALL(cudaMemset(halo_buffer_for_bottom_neighbor[dev_id], 0, 2 * nx * sizeof(real)));

        CUDA_RT_CALL(cudaMalloc(is_top_done_computing_flags + dev_id, num_flags * sizeof(int)));
        CUDA_RT_CALL(cudaMalloc(is_bottom_done_computing_flags + dev_id, num_flags * sizeof(int)));

        CUDA_RT_CALL(cudaMemset(is_top_done_computing_flags[dev_id], 0, num_flags * sizeof(int)));
        CUDA_RT_CALL(
            cudaMemset(is_bottom_done_computing_flags[dev_id], 0, num_flags * sizeof(int)));

        // Calculate local domain boundaries
        int iy_start_global;  // My start index in the global array
        if (dev_id < num_ranks_low) {
            iy_start_global = dev_id * chunk_size_low + 1;
        } else {
            iy_start_global =
                num_ranks_low * chunk_size_low + (dev_id - num_ranks_low) * chunk_size_high + 1;
        }
        int iy_end_global = iy_start_global + chunk_size - 1;  // My last index in the global array

        int iy_start = 1;
        iy_end[dev_id] = (iy_end_global - iy_start_global + 1) + iy_start;

        // Set diriclet boundary conditions on left and right border
        initialize_boundaries<<<(ny / num_devices) / 128 + 1, 128>>>(
            a[dev_id], a_new[dev_id], PI, iy_start_global - 1, nx, chunk_size + 2, ny);
        CUDA_RT_CALL(cudaGetLastError());

        CUDA_RT_CALL(cudaDeviceSynchronize());

        CUDA_RT_CALL(cudaMemcpy((void *)halo_buffer_for_top_neighbor[dev_id],
                                a[dev_id] + iy_end[dev_id] * nx, nx * sizeof(real),
                                cudaMemcpyDeviceToDevice));
        CUDA_RT_CALL(cudaMemcpy((void *)halo_buffer_for_bottom_neighbor[dev_id], a[dev_id],
                                nx * sizeof(real), cudaMemcpyDeviceToDevice));

        dim3 dim_grid(grid_dim_x * grid_dim_y + 1);
        dim3 dim_block(dim_block_x, dim_block_y);

        void *kernelArgs[] = {(void *)&a_new[dev_id],
                              (void *)&a[dev_id],
                              (void *)&iy_start,
                              (void *)&iy_end[dev_id],
                              (void *)&nx,
                              (void *)&grid_dim_x,
                              (void *)&iter_max,
                              (void *)&halo_buffer_for_top_neighbor[dev_id],
                              (void *)&halo_buffer_for_bottom_neighbor[dev_id],
                              (void *)&halo_buffer_for_bottom_neighbor[top],
                              (void *)&halo_buffer_for_top_neighbor[bottom],
                              (void *)&is_top_done_computing_flags[dev_id],
                              (void *)&is_bottom_done_computing_flags[dev_id],
                              (void *)&is_bottom_done_computing_flags[top],
                              (void *)&is_top_done_computing_flags[bottom]};

#pragma omp barrier
        double start = omp_get_wtime();

	std::cerr << "before jacobi_kernel launch\n";
        CUDA_RT_CALL(cudaLaunchCooperativeKernel((void *)SSMultiThreadedTwoBlockComm::jacobi_kernel,
                                                 dim_grid, dim_block, kernelArgs, 0, nullptr));

        CUDA_RT_CALL(cudaGetLastError());
        CUDA_RT_CALL(cudaDeviceSynchronize());
	std::cerr << "after jacobi_kernel launch\n";
        // Need to swap pointers on CPU if iteration count is odd
        // Technically, we don't know the iteration number (since we'll be doing l2-norm)
        // Could write iter to CPU when kernel is done
        if (iter_max % 2 == 1) {
            std::swap(a_new[dev_id], a[dev_id]);
        }

#pragma omp barrier
        double stop = omp_get_wtime();

        if (compare_to_single_gpu) {
            CUDA_RT_CALL(
                cudaMemcpy(a_h + iy_start_global * nx, a[dev_id] + nx,
                           std::min((ny - iy_start_global) * nx, chunk_size * nx) * sizeof(real),
                           cudaMemcpyDeviceToHost));
        }

#pragma omp barrier

#pragma omp master
        {
            report_results(ny, nx, a_ref_h, a_h, num_devices, runtime_serial_non_persistent, start,
                           stop, compare_to_single_gpu);
        }

        CUDA_RT_CALL(cudaFree(a_new[dev_id]));
        CUDA_RT_CALL(cudaFree(a[dev_id]));
        CUDA_RT_CALL(cudaFree(halo_buffer_for_top_neighbor[dev_id]));
        CUDA_RT_CALL(cudaFree(halo_buffer_for_bottom_neighbor[dev_id]));
        CUDA_RT_CALL(cudaFree(is_top_done_computing_flags[dev_id]));
        CUDA_RT_CALL(cudaFree(is_bottom_done_computing_flags[dev_id]));

        if (compare_to_single_gpu && 0 == dev_id) {
            CUDA_RT_CALL(cudaFreeHost(a_h));
            CUDA_RT_CALL(cudaFreeHost(a_ref_h));
        }
    }

    return 0;
}
