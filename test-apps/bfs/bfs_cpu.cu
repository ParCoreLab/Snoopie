#include <queue>
#include <iostream>
    
void print_queue(std::queue<int> q)
{
  while (!q.empty())
  {
    std::cout << q.front() << " ";
    q.pop();
  }
  std::cout << std::endl;
}


void bfs_sequential(
    graph_t *graph,
    int start_vertex, 
    int *result)
{
    bool *visited = new bool[graph->num_vertices];
    fill_n(visited, graph->num_vertices, 0);
    visited[start_vertex] = true;
    //int start_vertex_2 = start_vertex + graph->num_vertices/2 + 1;

    //visited[start_vertex_2] = true;

    fill_n(result, graph->num_vertices, MAX_DIST);
    result[start_vertex] = 0;
    //result[start_vertex_2] = 0;

    queue<int> next_vertices;
    next_vertices.push(start_vertex);
    //next_vertices.push(start_vertex_2);
    int max_depth = 0;

    while (!next_vertices.empty())
    {
        //printf("Iter [%d]=", iter);
        //print_queue(next_vertices);
        int vertex = next_vertices.front();
        next_vertices.pop();

        for (
            int n = graph->v_adj_begin[vertex]; 
            n < graph->v_adj_begin[vertex] + graph->v_adj_length[vertex]; 
            n++)
        {
            int neighbor = graph->v_adj_list[n];

            if (!visited[neighbor])
            {
                visited[neighbor] = true;
                result[neighbor] = result[vertex] + 1;
                if (result[neighbor] > max_depth) max_depth = result[neighbor];
                next_vertices.push(neighbor);
            }
        }
    }

    int max_size = 0;

    // for (int i = 0; i < max_depth; i++) 
    // {
    //     int *lvl = new int[graph->num_vertices];
    //     int idx = 0;
    //     for (int v = 0; v < graph->num_vertices; v++) 
    //     {
    //         if (result[v] == i) 
    //         {
    //             lvl[idx] = v;
    //             idx++;
    //         }
    //     }
    //     if (idx > max_size) max_size = idx;
    //     std::cout << "Depth " << i << " | Size " << idx << std::endl;//": ";
	// /*
    //     for (int lv = 0; lv < idx-1; lv++) 
    //     {
    //         std::cout << lvl[lv] << ", ";
    //     }
    //     std::cout << lvl[idx-1] << std::endl << std::endl;

    //     for (int lv = 0; lv < idx; lv++) 
    //     {
    //         int v = lvl[lv];
    //         int n = graph->v_adj_begin[v]; 
    //         std::cout << v << " | size = " << graph->v_adj_length[v] << ": ";
    //         while (n < graph->v_adj_begin[v] + graph->v_adj_length[v] - 1)
    //         {
    //             int neighbor = graph->v_adj_list[n];
    //             std::cout << neighbor << ", "; 
    //             n++;
    //         }
    //         std::cout << graph->v_adj_list[n] << std::endl;
    //     }
    //     std::cout << std::endl;*/
    //     delete[] lvl;
    // }
    delete[] visited;
    if (verbose) std::cout << "Start=" << start_vertex << " MaxDepth=" << max_depth << " MaxSize=" << max_size << endl;
}
