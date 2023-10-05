// Accelerating large graph algorithms on the GPU using CUDA
// http://dl.acm.org/citation.cfm?id=1782200

__global__ void kernel_cuda_frontier(int *v_adj_list, int *v_adj_begin,
                                     int *v_adj_length, int num_vertices,
                                     int *result, bool *updated, bool *frontier,
                                     bool *still_running, bool *visited) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int num_threads = blockDim.x * gridDim.x;

  for (int v = 0; v < num_vertices; v += num_threads) {
    int vertex = v + tid;

    if (vertex < num_vertices && frontier[vertex]) {
      frontier[vertex] = false;

      for (int n = 0; n < v_adj_length[vertex]; n++) {
        int neighbor = v_adj_list[v_adj_begin[vertex] + n];

        if (!visited[neighbor]) {
          result[neighbor] = result[vertex] + 1;
          updated[neighbor] = true;
        }
      }
    }
  }
}

__global__ void kernel_cuda_frontier_update_flags(int num_vertices,
                                                  bool *still_running,
                                                  bool *updated, bool *frontier,
                                                  bool *visited) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int num_threads = blockDim.x * gridDim.x;

  for (int v = 0; v < num_vertices; v += num_threads) {
    int vertex = v + tid;

    if (vertex < num_vertices && updated[vertex]) {
      frontier[vertex] = true;
      updated[vertex] = false;
      visited[vertex] = true;
      *still_running = true;
    }
  }
}

int bfs_cuda_frontier(int *v_adj_list, int *v_adj_begin, int *v_adj_length,
                      int num_vertices, int num_edges, int start_vertex,
                      int *result) {
  int *k_v_adj_list;
  int *k_v_adj_begin;
  int *k_v_adj_length;
  int *k_result;
  bool *k_updated;
  bool *k_still_running;
  bool *k_frontier;
  bool *k_visited;

  int kernel_runs = 0;

  bool *updated = new bool[num_vertices];
  fill_n(updated, num_vertices, false);

  bool *visited = new bool[num_vertices];
  fill_n(visited, num_vertices, false);
  visited[start_vertex] = true;

  bool *frontier = new bool[num_vertices];
  fill_n(frontier, num_vertices, false);
  frontier[start_vertex] = true;

  fill_n(result, num_vertices, MAX_DIST);
  result[start_vertex] = 0;

  bool *still_running = new bool[1];
  bool false_value = false;

  cudaMalloc(&k_v_adj_list, sizeof(int) * num_edges);
  cudaMalloc(&k_v_adj_begin, sizeof(int) * num_vertices);
  cudaMalloc(&k_v_adj_length, sizeof(int) * num_vertices);
  cudaMalloc(&k_result, sizeof(int) * num_vertices);
  cudaMalloc(&k_updated, sizeof(bool) * num_vertices);
  cudaMalloc(&k_frontier, sizeof(bool) * num_vertices);
  cudaMalloc(&k_still_running, sizeof(bool) * 1);
  cudaMalloc(&k_visited, sizeof(bool) * num_vertices);

  cudaMemcpy(k_v_adj_list, v_adj_list, sizeof(int) * num_edges,
             cudaMemcpyHostToDevice);
  cudaMemcpy(k_v_adj_begin, v_adj_begin, sizeof(int) * num_vertices,
             cudaMemcpyHostToDevice);
  cudaMemcpy(k_v_adj_length, v_adj_length, sizeof(int) * num_vertices,
             cudaMemcpyHostToDevice);
  cudaMemcpy(k_result, result, sizeof(int) * num_vertices,
             cudaMemcpyHostToDevice);
  cudaMemcpy(k_updated, updated, sizeof(bool) * num_vertices,
             cudaMemcpyHostToDevice);
  cudaMemcpy(k_visited, visited, sizeof(bool) * num_vertices,
             cudaMemcpyHostToDevice);
  cudaMemcpy(k_frontier, frontier, sizeof(bool) * num_vertices,
             cudaMemcpyHostToDevice);

  // --- START MEASURE TIME ---

  struct timeval t1, t2;
  gettimeofday(&t1, NULL);

  *still_running = false;

  do {
    cudaMemcpy(k_still_running, &false_value, sizeof(bool) * 1,
               cudaMemcpyHostToDevice);

    kernel_cuda_frontier<<<BLOCKS, THREADS>>>(
        k_v_adj_list, k_v_adj_begin, k_v_adj_length, num_vertices, k_result,
        k_updated, k_frontier, k_still_running, k_visited);

    cudaDeviceSynchronize();

    kernel_cuda_frontier_update_flags<<<BLOCKS, THREADS>>>(
        num_vertices, k_still_running, k_updated, k_frontier, k_visited);

    cudaDeviceSynchronize();

    kernel_runs++;

    cudaMemcpy(still_running, k_still_running, sizeof(bool) * 1,
               cudaMemcpyDeviceToHost);
  } while (*still_running);

  cudaDeviceSynchronize();

  gettimeofday(&t2, NULL);
  long long time = get_elapsed_time(&t1, &t2);

  if (report_time)
    printf("%s, %i, %i, %s, %i, %i, N/A, N/A, %lld\n", filename, 1,
           start_vertex, __FILE__, BLOCKS, THREADS, time);
  // --- END MEASURE TIME ---

  cudaMemcpy(result, k_result, sizeof(int) * num_vertices,
             cudaMemcpyDeviceToHost);

  cudaFree(k_v_adj_list);
  cudaFree(k_v_adj_begin);
  cudaFree(k_v_adj_length);
  cudaFree(k_result);
  cudaFree(k_still_running);
  cudaFree(k_updated);
  cudaFree(k_frontier);

  // printf("%i kernel runs\n", kernel_runs);

  return time;
}
