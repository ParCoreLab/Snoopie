// Accelerating large graph algorithms on the GPU using CUDA
// http://dl.acm.org/citation.cfm?id=1782200
#include "cuda_wrapper.hpp"
#include <cooperative_groups.h>

#define CACHE_LINE_SIZE 16 // in terms of ints
#define F_DONE 0 * CACHE_LINE_SIZE
#define F_PULL 1 * CACHE_LINE_SIZE
#define F_PUSH 10 * CACHE_LINE_SIZE
#define F_DISCOVERED 2 * CACHE_LINE_SIZE
#define F_PULL_EXTRA 3 * CACHE_LINE_SIZE
#define F_SIZE 4 * CACHE_LINE_SIZE
#define F_ITER 5 * CACHE_LINE_SIZE
#define F_QUEUE_IN 6 * CACHE_LINE_SIZE
#define F_QUEUE_OUT 7 * CACHE_LINE_SIZE
#define F_WORKER_SIZE 8 * CACHE_LINE_SIZE
#define F_COMM 9 * CACHE_LINE_SIZE
#define FLAGS_SIZE 16 * CACHE_LINE_SIZE
#define CUDA_CHECK(call)                                                       \
  if ((call) != cudaSuccess) {                                                 \
    cudaError_t err = cudaGetLastError();                                      \
    printf("CUDA error calling method \"" #call "\" - err: %s\n",              \
           cudaGetErrorString(err));                                           \
  }

namespace cg = cooperative_groups;

void checkAndSetP2Paccess(int numGPUs) {
  for (int i = 0; i < numGPUs; i++) {
    cudaSetDevice(i);

    for (int j = 0; j < numGPUs; j++) {
      int access;
      if (i != j) {
        cudaDeviceCanAccessPeer(&access, i, j);
        if (!access)
          printf("Device=%d CANNOT Access Peer Device=%d\n", i, j);
        // printf("Device=%d %s Access Peer Device=%d\n", i, access ? "CAN" :
        // "CANNOT", j);
        if (access)
          cudaDeviceEnablePeerAccess(j, 0);
      }
    }
  }
}

__device__ bool done;
__device__ bool offloaded;
__device__ bool pushing;
__device__ bool pulling;
__device__ bool idle;
__device__ int extra_size;
__device__ int push_size;
__device__ int pull_size;
__device__ int offset;

// update flags after pulling
__device__ void update_pull_flags(int device_id, int N_GPU, volatile int *flags,
                                  volatile int *flags_prev,
                                  volatile int *flags_next) {
  flags[F_PULL] = 0;
  if (device_id != N_GPU - 1) {
    flags_next[F_PULL] = 1;
  } else {
    flags_next[F_PUSH] = 1;
  }
  pulling = false;
}

// update flags after pushing
__device__ void update_push_flags(int device_id, int N_GPU,
                                  int discovered_vertices_num,
                                  volatile int *flags, volatile int *flags_prev,
                                  volatile int *flags_next) {
  flags[F_PUSH] = 0;
  int discovered_total = flags_prev[F_DISCOVERED] + discovered_vertices_num;
  if (device_id != N_GPU - 1) {
    flags[F_DISCOVERED] = discovered_total;
    flags_next[F_PUSH] = 1;
  } else {
    flags_next[F_PULL] = 1;
    if (flags_prev[F_DISCOVERED] + discovered_vertices_num == 0) {
      flags[F_DONE] += 1;
    }
  }
  // printf("%d device started GLOBAL PUSH of %d elements. Discovered: %d\n",
  // device_id, push_size, discovered_vertices_num);
  pushing = false;
}

// pull everything, without offloading
__device__ void pull_all(int vert_start, int vert_end, int *global_queue,
                         int *global_queue_prev, int *local_in_queue, int N_GPU,
                         int tid, int num_threads, int *result) {
  for (int v = 0; v < pull_size; v += num_threads) {
    if (v + tid < pull_size) {
      int index = (v + tid) * 2 + N_GPU + 1;
      int vertex = global_queue_prev[index];
      int vertex_depth = global_queue_prev[index + 1];

      if (vertex >= vert_start && vertex < vert_end) {
        if (atomicMin(&result[vertex], vertex_depth) > vertex_depth)
        // if (result[vertex] > vertex_depth)
        {
          // result[vertex] = vertex_depth;
          int position = atomicAdd(local_in_queue, 1);
          local_in_queue[position + 1] = vertex;
        }
      } else {
        int position = atomicAdd(global_queue, 1) * 2 + N_GPU + 1;
        global_queue[position] = vertex;
        global_queue[position + 1] = vertex_depth;
      }
    }
  }
}

// pull up to the threshold limit
__device__ void pull_with_offload(int vert_start, int vert_end,
                                  int *global_queue, int *global_queue_prev,
                                  int *local_in_queue, int threshold, int N_GPU,
                                  int tid, int num_threads) {
  for (int v = 0; v < pull_size; v += num_threads) {
    if (v + tid < pull_size) {
      int vertex = global_queue_prev[v + tid + N_GPU + 1];
      if (vertex < vert_end && local_in_queue[0] < threshold) // && !offloaded)
      {
        int position = atomicAdd(local_in_queue, 1);
        if (position >= threshold) {
          offloaded = true;
          position = atomicAdd(global_queue, 1);
          global_queue[position + N_GPU + 1] = vertex;
        } else {
          local_in_queue[position + 1] = vertex;
        }
      } else {
        int position = atomicAdd(global_queue, 1);
        global_queue[position + N_GPU + 1] = vertex;
      }
    }
  }
}

// work iteration
__device__ void bfs_iteration(int *v_adj_list, int *v_adj_begin,
                              int *v_adj_length, int vert_start, int vert_end,
                              int *result, int *queue_to_push,
                              int *local_in_queue, int *local_out_queue,
                              int tid, int num_threads) {
  int loc_insize = local_in_queue[0];

  for (int v = 0; v < loc_insize; v += num_threads) {
    if (v + tid < loc_insize) {
      int vertex = local_in_queue[v + tid + 1];
      int child_depth = result[vertex] + 1;

      for (int n = 0; n < v_adj_length[vertex]; n++) {
        int neighbor = v_adj_list[v_adj_begin[vertex] + n];

        if (atomicMin(&result[neighbor], child_depth) > child_depth)
        // if (result[neighbor] > child_depth)
        {
          // result[neighbor] = child_depth;
          // Add to queue (atomicAdd returns original value)
          if (neighbor >= vert_start && neighbor < vert_end) {
            int position = atomicAdd(local_out_queue, 1);
            local_out_queue[position + 1] = neighbor;
            // if (tid % 4 == 0) {
            int position_push = atomicAdd(queue_to_push, 1);
            queue_to_push[position_push + 1] = neighbor;
            //}
          } else {
            // if the vertex doesn't belong to this GPU
            int position_push = atomicAdd(queue_to_push, 1);
            queue_to_push[position_push + 1] = neighbor;
          }
        }
      }
    }
  }
}

__global__ void
kernel_workq(int *v_adj_list, int *v_adj_begin, int *v_adj_length,
             int num_all_vert, int vert_start, int vert_end, int *result,
             int *global_queue, int *global_queue_prev, int *queue_to_push,
             int *local_in_queue, int *local_out_queue, int off_threshold,
             int comm_threshold, int device_id, int N_GPU, volatile int *flags,
             volatile int *flags_prev, volatile int *flags_next,
             volatile int **all_flags, int prev, int next, int *metadata) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int num_threads = blockDim.x * gridDim.x;
  cg::grid_group grid = cg::this_grid(); // this device's grid

  // if (tid == 0) printf("Started device %d kernel working on
  // vertices:[%d:%d]\n", device_id, vert_start, vert_end);

  // repeat until termination
  while (true) {
    if (tid == 0) {
      if (flags[F_DONE] ==
          2 /*|| metadata[0] == 6*/) // this handles termination without
                                     // unsyncing
      {
        all_flags[next][F_DONE] = 2;
        done = true;
      }

      // printf("%d device: %d elements. Iter %d\n", device_id,
      // local_in_queue[0], metadata[0]);

      // for (int i = 0; i < local_in_queue[0]; i++) {
      //     printf("v%d ", local_in_queue[1+i]);
      // }
      // printf("\n");

      if (local_in_queue[0] != 0)
        metadata[0] += 1;

      local_out_queue[0] = 0; // empty the local queue

      // check if it's my turn to pull from the global queue
      if (flags[F_PULL] == 1) {
        pull_size = *global_queue_prev;
        if (device_id != 0)
          idle = false;
        // if (pull_size == 1 || pull_size == 2) {
        if (pull_size > 0) {
          metadata[1] += 1;
          metadata[4] += pull_size;
          pulling = true;
        } else {
          global_queue_prev[0] = 0;
          update_pull_flags(device_id, N_GPU, flags, all_flags[prev],
                            all_flags[next]);
        }
        // printf("%d device: PULL %d elements. Current glob size %d\n",
        // device_id, pull_size, global_queue[0]);
      }

      // this reduces the amount of communications
      // if (device_id != 0 || flags[F_COMM] == 1) {
      // push to the global queue
      if (flags[F_PUSH] == 1) {
        metadata[3] += 1;
        // idle = false;
        // flags[F_COMM] = 0;
        push_size = queue_to_push[0];
        /* NOTE: no need for atomic -> offset = atomicAdd(global_queue,
        push_size); because only one device and 1 thread can be pushing at a
        time */
        offset = global_queue[0] * 2;
        if (push_size > 0) {
          metadata[2] += 1;
          global_queue[0] += push_size;
          pushing = true;
        } else {
          update_push_flags(device_id, N_GPU,
                            *global_queue + *local_in_queue + *local_out_queue,
                            flags, all_flags[prev], all_flags[next]);
        }
      }
      //}

      /*if (flags[F_PULL_EXTRA] == 1)
      {
          flags[F_PULL_EXTRA] = 0;
          extra_size =
          pulling_extra = true;
      }*/
    }

    grid.sync(); // ensure all the threads are in/out of pull/push stage OR
                 // termination stage

    if (pulling) {
      /*if (threshold > 0)
          pull_with_offload(vert_start, vert_end, global_queue,
      global_queue_prev, local_in_queue, threshold, N_GPU, tid, num_threads);
      else
          pull_all(vert_start, vert_end, global_queue, global_queue_prev,
              local_in_queue, N_GPU, tid, num_threads, result);*/
      pull_all(vert_start, vert_end, global_queue, global_queue_prev,
               local_in_queue, N_GPU, tid, num_threads, result);

      grid.sync(); // needed to ensure the local_in_queue is ready

      if (tid == 0) {
        if (offloaded)
          local_in_queue[0] = off_threshold;
        // printf("GPU%d, %d, %d\n", device_id, local_in_queue[0],
        // global_queue[0]);
        offloaded = false;
        global_queue_prev[0] = 0;
        update_pull_flags(device_id, N_GPU, flags, all_flags[prev],
                          all_flags[next]);
      }
    }

    if (pushing) {
      // push all the elements from the temporary queue to the global queue
      for (int v = 0; v < push_size; v += num_threads) {
        if (v + tid < push_size) {
          int vertex = queue_to_push[v + tid + 1];
          int index = offset + N_GPU + 1 + 2 * (v + tid);
          global_queue[index] = vertex;
          global_queue[index + 1] = result[vertex];
        }
      }

      grid.sync(); // needed to make sure the next GPU can pull safely

      if (tid == 0) {
        // for (int i = 0; i < global_queue[0]; i++) {
        //     printf("v%d=%d ", global_queue[1+N_GPU+i*2],
        //     global_queue[2+N_GPU+i*2]);
        // }
        // printf("\n");
        queue_to_push[0] = 0; // empty the temporary queue
        update_push_flags(device_id, N_GPU,
                          *global_queue + *local_in_queue + *local_out_queue,
                          flags, all_flags[prev], all_flags[next]);
      }
    }

    /*if (pulling_extra)
    {
        for (int v = 0; v < push_size; v += num_threads)
        {
            if (v + tid < push_size)
            {

    }*/

    bfs_iteration(v_adj_list, v_adj_begin, v_adj_length, vert_start, vert_end,
                  result, queue_to_push, local_in_queue, local_out_queue, tid,
                  num_threads);

    grid.sync(); // needed to ensure the local_out_queue is ready

    // if (tid == 0 && !idle && local_out_queue[0] <= comm_threshold) {
    //     all_flags[0][F_COMM] = 1;
    //     //if (device_id != 0) {
    //         //idle = true;
    //     //}
    // }

    // swap pointers
    int *tmp = local_in_queue;
    local_in_queue = local_out_queue;
    local_out_queue = tmp;

    if (done)
      return; // terminate
  }
}

int workq_ring(int *adj_list, int *adj_begin, int *adj_length, int num_vertices,
               int num_edges, int start_vertex, int *res) {
  checkAndSetP2Paccess(N_GPU);

  int num_vert_per_device[N_GPU];
  fill_n(num_vert_per_device, N_GPU, (num_vertices / N_GPU));
  num_vert_per_device[0] += num_vertices % N_GPU;

  int *v_adj_list[N_GPU];
  int *v_adj_begin[N_GPU];
  int *v_adj_length[N_GPU];

  int *loc_in_queue[N_GPU];
  int *loc_out_queue[N_GPU];
  int *glob_queue[N_GPU];
  int *queue_to_push[N_GPU];
  int *offload_buffer[N_GPU];
  int *result[N_GPU];
  int *metadata[N_GPU];
  int *meta;
  volatile int *flags[N_GPU];
  int *flags_h;
  CUDA_CHECK(cudaMallocHost(&flags_h, sizeof(int) * FLAGS_SIZE));
  CUDA_CHECK(cudaMallocHost(&meta, sizeof(int) * FLAGS_SIZE));
  volatile int **all_flags[N_GPU];

  int zero_value = 0;

  fill_n(res, num_vertices, MAX_DIST);
  fill_n(flags_h, FLAGS_SIZE, 0);
  res[start_vertex] = 0;

  int *input_queue_size = new int;
  int first_queue[] = {1, start_vertex};
  *input_queue_size = 1;

  for (int device = 0; device < N_GPU; device++) {
    CUDA_CHECK(cudaSetDevice(device));
    CUDA_CHECK(cudaMalloc(&v_adj_list[device], sizeof(int) * num_edges));
    CUDA_CHECK(cudaMalloc(&v_adj_begin[device], sizeof(int) * num_vertices));
    CUDA_CHECK(cudaMalloc(&v_adj_length[device], sizeof(int) * num_vertices));
    CUDA_CHECK(cudaMemcpy(v_adj_list[device], adj_list, sizeof(int) * num_edges,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(v_adj_begin[device], adj_begin,
                          sizeof(int) * num_vertices, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(v_adj_length[device], adj_length,
                          sizeof(int) * num_vertices, cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&flags[device], sizeof(int) * FLAGS_SIZE));
    CUDA_CHECK(cudaMemcpy((void *)flags[device], (void *)flags_h,
                          sizeof(int) * FLAGS_SIZE, cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&loc_in_queue[device],
                          sizeof(int) * (num_vert_per_device[device] + 1)));
    CUDA_CHECK(cudaMalloc(&loc_out_queue[device],
                          sizeof(int) * (num_vert_per_device[device] + 1)));
    CUDA_CHECK(cudaMalloc(&glob_queue[device],
                          sizeof(int) * (num_vertices * 2 + 1 + N_GPU)));
    CUDA_CHECK(cudaMalloc(&queue_to_push[device],
                          sizeof(int) * (num_vertices * 2 + 1)));
    CUDA_CHECK(cudaMemcpy(glob_queue[device], &zero_value, sizeof(int) * 1,
                          cudaMemcpyHostToDevice));
    // CUDA_CHECK(cudaMemcpy(loc_in_queue[device], &zero_value, sizeof(int) * 1,
    // cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(loc_out_queue[device], &zero_value, sizeof(int) * 1,
                          cudaMemcpyHostToDevice));

    CUDA_CHECK(
        cudaMalloc(&offload_buffer[device], sizeof(int) * (num_vertices + 1)));
    CUDA_CHECK(cudaMemcpy(offload_buffer[device], &zero_value, sizeof(int) * 1,
                          cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&result[device], sizeof(int) * num_vertices));
    CUDA_CHECK(cudaMemcpy(result[device], res, sizeof(int) * num_vertices,
                          cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&metadata[device], sizeof(int) * 5));

    if (device == 0) { // place the flags allowing for device 0 to start
      flags_h[F_PULL] = 1;
      flags_h[F_PUSH] = 0;
      // flags_h[F_COMM] = 1;
      CUDA_CHECK(cudaMemcpy(loc_in_queue[device], first_queue, sizeof(int) * 2,
                            cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy((void *)flags[device], (void *)flags_h,
                            sizeof(int) * FLAGS_SIZE, cudaMemcpyHostToDevice));
      // flags_h[F_COMM] = 0;
    } else {
      flags_h[F_PULL] = 0;
      flags_h[F_PUSH] = 0;
      CUDA_CHECK(cudaMemcpy((void *)flags[device], (void *)flags_h,
                            sizeof(int) * FLAGS_SIZE, cudaMemcpyHostToDevice));
    }
  }

  for (int device = 0; device < N_GPU; device++) {
    CUDA_CHECK(cudaMalloc(&all_flags[device], sizeof(volatile int *) * N_GPU));
    CUDA_CHECK(cudaMemcpy(all_flags[device], flags,
                          sizeof(volatile int *) * N_GPU,
                          cudaMemcpyHostToDevice));
  }

  int blocks = BLOCKS;
  int threads = THREADS;
  int off_threshold = blocks * threads * OFF_THRESH;
  int comm_threshold = (COMM_THRESH == -1) ? num_vertices : COMM_THRESH;

  cudaStream_t streams[N_GPU];

  // if (verbose) printf("Time start \n");
  // --- START MEASURE TIME ---
  struct timeval t1, t2;
  gettimeofday(&t1, NULL);

  int v_start = 0;
  int v_end = 0;

  for (int device = 0; device < N_GPU; device++) {
    CUDA_CHECK(cudaSetDevice(device));
    CUDA_CHECK(cudaStreamCreate(&(streams[device])));

    v_start = v_end;
    v_end += num_vert_per_device[device];

    int prev = (device == 0) ? N_GPU - 1 : device - 1;
    int next = (device == N_GPU - 1) ? 0 : device + 1;

    void *kernelArgs[] = {(void *)&v_adj_list[device],
                          (void *)&v_adj_begin[device],
                          (void *)&v_adj_length[device],
                          (void *)&num_vertices,
                          (void *)&v_start,
                          (void *)&v_end,
                          (void *)&result[device],
                          (void *)&glob_queue[device],
                          (void *)&glob_queue[prev],
                          (void *)&queue_to_push[device],
                          (void *)&loc_in_queue[device],
                          (void *)&loc_out_queue[device],
                          (void *)&off_threshold,
                          (void *)&comm_threshold,
                          (void *)&device,
                          (void *)&N_GPU,
                          (void *)&flags[device],
                          (void *)&flags[prev],
                          (void *)&flags[next],
                          (void *)&all_flags[device],
                          (void *)&prev,
                          (void *)&next,
                          (void *)&metadata[device]};

    CUDA_CHECK(cudaLaunchCooperativeKernel(
        (void *)kernel_workq, blocks, threads, kernelArgs, 0, streams[device]));
  }

  // if (verbose) printf("Sync start \n");

  for (int device = N_GPU - 1; device >= 0; device--) {
    CUDA_CHECK(cudaSetDevice(device));
    // CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaStreamSynchronize(streams[device]));
    // if (verbose) printf("Sync GPU %d done \n", device);
  }
  gettimeofday(&t2, NULL);

  long long time = get_elapsed_time(&t1, &t2);

  if (report_time)
    printf("%s, %i, %i, %s, %i, %i, %.2f, %i, %lld", filename, N_GPU,
           start_vertex, __FILE__, BLOCKS, THREADS, OFF_THRESH, COMM_THRESH,
           time);

  for (int i = 0; i < N_GPU; i++) {
    CUDA_CHECK(cudaMemcpy(&meta[0], metadata[i], 5 * sizeof(int),
                          cudaMemcpyDeviceToHost));
    printf(", %i, %i, %i, %i", meta[0], meta[1], meta[2], meta[4]);
  }
  printf(", %i\n", meta[3]);

  // --- END MEASURE TIME ---
  CUDA_CHECK(cudaSetDevice(0));
  // copy the result to host to verify
  int res_offset = 0;
  for (int i = 0; i < N_GPU; i++) {
    CUDA_CHECK(cudaMemcpy(&res[res_offset], &result[i][res_offset],
                          num_vert_per_device[i] * sizeof(int),
                          cudaMemcpyDeviceToHost));
    res_offset += num_vert_per_device[i];
  }

  // free the memory
  for (int device = 0; device < N_GPU; device++) {
    CUDA_CHECK(cudaSetDevice(device));
    CUDA_CHECK(cudaFree(result[device]));
    CUDA_CHECK(cudaFree(v_adj_list[device]));
    CUDA_CHECK(cudaFree(v_adj_begin[device]));
    CUDA_CHECK(cudaFree(v_adj_length[device]));
    CUDA_CHECK(cudaFree(glob_queue[device]));
    CUDA_CHECK(cudaFree(loc_in_queue[device]));
    CUDA_CHECK(cudaFree(loc_out_queue[device]));
    CUDA_CHECK(cudaFree((void *)flags[device]));
    CUDA_CHECK(cudaFree(metadata[device]));
    CUDA_CHECK(cudaDeviceReset());
  }
  return time;
}
