#include <cuda.h>
#include <limits>
#include <stdlib.h>
#include <libgen.h>
#include <stdio.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <string.h>
#include <algorithm>
#include <vector>
#include <map>
#include <math.h>
#include <iterator>
#include <sys/time.h>
#include <assert.h>
#include <unistd.h>

using namespace std;

int BLOCKS = 0;
int THREADS = 0;
int N_GPU = 0;
float OFF_THRESH = 0.0;
int COMM_THRESH = -1;
int runs;
char *filename;
int report_time = true;
int verbose = false;
int metis = false;

#define MAX_DIST 1073741824
#define MAX_KERNEL_RUNS 2048
#define RT_STUDY_NUM_V 5

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

struct graph_t
{
    int *v_adj_list;
    int *v_adj_begin;
    int *v_adj_length;  
    int num_vertices;    
    int num_edges;
    int *split_offsets;  
};

/* in microseconds (us) */
long get_elapsed_time(struct timeval *begin, struct timeval *end)
{
    return (end->tv_sec - begin->tv_sec) * 1000 * 1000
            + (end->tv_usec - begin->tv_usec); /// 1000.0;
}


#include "bfs_cpu.cu"
// #include "multi_offload.cu"
#include "workq_ring.cu"
// #include "multi_split.cu"
// #include "single_cg_frontier.cu"
// #include "multi_split_queue.cu"
#include "bfs_cuda_simple.cu"
#include "bfs_cuda_frontier.cu"
#include "bfs_cuda_frontier_numbers.cu"
// #include "bfs_cuda_frontier_queue.cu"


typedef int (* bfs_func)(int*, int*, int*, int, int, int, int*);

int run_bfs(bfs_func func, graph_t *graph, int start_vertex, int *expected, int runs)
{
    int *result = new int[graph->num_vertices];

    int runtime = 1073741824;

    for (int i = 0; i < runs; i++)
    {
        // Reset result array
        memset(result, 0, sizeof(int) * graph->num_vertices);

        int next_time = func(
            graph->v_adj_list, 
            graph->v_adj_begin, 
            graph->v_adj_length, 
            graph->num_vertices, 
            graph->num_edges,
            start_vertex, 
            result);

        runtime = min(next_time, runtime);
        
        if (!equal(result, result + graph->num_vertices, expected))
        {
            int errs = 0;
            for (int m = 0; m < graph->num_vertices; m++)
            {
                if (expected[m] != result[m])
                {
                    errs++;
                    if (errs < 10) printf("Vertex %d: %d vs %d\n", m, result[m], expected[m]);
                }
            }
            printf("Completed with %d errors\n", errs);
            // Wrong result
            return -1;
        }
    }

    free(result);

    return runtime;
}

const bfs_func bfs_functions[] = {
    //&multi_offload,
    &workq_ring,
    //&multi_split,
    //&single_cg_frontier, 
    //&multi_split_queue,
    &bfs_cuda_simple, 
    &bfs_cuda_frontier, 
    &bfs_cuda_frontier_numbers, 
    //&bfs_cuda_frontier_queue
    };

string bfs_names[] = {
    //"multi_offload",
    "workq_ring",
    //"multi_split",
    //"single_cg_frontier",
    //"multi_split_queue",
    "bfs_cuda_simple",
    "bfs_cuda_frontier", 
    "bfs_cuda_frontier_numbers", 
    //"bfs_cuda_frontier_scan"
    };
   

void run_all_bfs(graph_t *graph, int start_vertex)
{
     
    int num_bfs = sizeof(bfs_functions) / sizeof(*bfs_functions);
    double *runtime = new double[num_bfs]();
    bool *wrong_result = new bool[num_bfs]();

    int range_from, range_to;
    int stride = 1;
    if (start_vertex == -1)
    {
        // Run for all start vertices
        range_from = 0;
        range_to = graph->num_vertices;
    }
    else if (start_vertex == -2)
    {
        // Run for RT_STUDY_NUM_V many vertices
        range_from = 0;
        range_to = graph->num_vertices;
        stride = graph->num_vertices / RT_STUDY_NUM_V;
    }
    else
    {
        range_from = start_vertex;
        range_to = start_vertex + 1;
    }

    
    int *expected = new int[graph->num_vertices];

    for (int vertex = range_from; vertex < range_to; vertex+=stride)
    {
        if (verbose) printf("vertex=%d\n", vertex);
        bfs_sequential(graph, vertex, expected);

        for (int i = 0; i < num_bfs; i++)
        {
            int next_runtime = run_bfs(bfs_functions[i], graph, vertex, expected, runs);

            if (next_runtime == -1)
            {
                // Wrong result
                wrong_result[i] = true;
            }
            else
            {
                runtime[i] += next_runtime;
            }
        }
    }

    for (int i = 0; i < num_bfs; i++)
    {
        double avg_runtime = runtime[i] / (range_to - range_from);

        if (verbose) 
        {
            if (!wrong_result[i])
            {
                printf("%s,%s,%i,%i,%i,%f\n", filename, bfs_names[i].c_str(), N_GPU, BLOCKS, THREADS, avg_runtime);
            }
            else
            {
                printf("%s,%s,%i,%i,%i,-1\n", filename, bfs_names[i].c_str(), N_GPU, BLOCKS, THREADS);
            }
        }
    }

    free(expected);
}


int* read_part(graph_t *graph)
{

    ifstream infile(string(filename).substr(0, string(filename).length()-4)  + ".g.part." + to_string(N_GPU));
    if (!infile.is_open()) 
    { 
        printf("Partition file did not open\n");
        exit(1); 
    }

    int num_vertices = graph->num_vertices;
    int *v_owner = new int[num_vertices];
    int index = 0;
    int owner;

    while (infile >> owner)
    {
        v_owner[index++] = owner;
    }

    infile.clear();
    return v_owner;
}



graph_t* metis_part(graph_t *graph_orig)
{

    ifstream infile(string(filename) + ".part." + to_string(N_GPU));
    
    if (!infile.is_open()) 
    { 
        printf("Partition file did not open\n");
        exit(1); 
    }
    int num_edges = graph_orig->num_edges;
    int num_vertices = graph_orig->num_vertices;

    int *v_adj_begin = new int[num_vertices];
    int *v_adj_length = new int[num_vertices];
    int *v_adj_list = new int[num_edges];
    int *split_offsets = new int[N_GPU];

    int vertex = 0;
    int vertex_index = 0;
    int edge_index = 0;
    int owner = -1;
    for (int device = 0; device < N_GPU; device++) {
        //printf("beg[0] = %d\n", v_adj_begin[0]);
        split_offsets[device] = vertex_index;

        while (infile >> owner)
        {
            if (owner == device) 
            {
                if (vertex_index == 0) printf("Orig length = %d\n", graph_orig->v_adj_length[vertex]);
                if (vertex_index == 0) printf("This offset = %d\n", edge_index);
                if (vertex_index == 0) printf("This orig vertex = %d\n", vertex);
                v_adj_length[vertex_index] = graph_orig->v_adj_length[vertex];
                v_adj_begin[vertex_index] = edge_index;
                for (int i = graph_orig->v_adj_begin[vertex]; 
                    i < graph_orig->v_adj_begin[vertex]+v_adj_length[vertex_index]; i++) 
                {
                    v_adj_list[edge_index] = graph_orig->v_adj_list[i];
                    edge_index++;
                }
                vertex_index++;
            }
            vertex++;
        }

        infile.clear();
        infile.seekg(0);
        vertex = 0;
    }
    printf("beg[0] = %d\n", v_adj_begin[0]);

    for (int i = 0; i < 200; i++) {
        printf("%d ", v_adj_list[i]);
    }
    printf("\n");

    for (int i = 0; i < 60; i++) {
        printf("%d|", v_adj_begin[i]);
        printf("%d ", v_adj_length[i]);
    }
    printf("\n");

    for (int i = 0; i < N_GPU; i++) {
        printf("%d ", split_offsets[i]);
    }
    printf("\n");


    graph_t *graph = new graph_t;
    graph->v_adj_list = v_adj_list;
    graph->v_adj_begin = v_adj_begin;
    graph->v_adj_length = v_adj_length;
    graph->num_vertices = num_vertices;
    graph->num_edges = num_edges;   
    graph->split_offsets = split_offsets;   
    return graph;
}


graph_t* load_ggraph()
{

    // Find number of vertices
    // printf("Reading input file\n");
    ifstream infile(filename);
    
    if (!infile.is_open()) 
    { 
        printf("File did not open\n");
        exit(1); 
    }

    //filename = basename(filename);

    int from, to;
    int edge_index = 0;
    int num_edges = 0;
    int num_vertices = 0;

    string line;
    //skip the comments, num vertices and edges line
    //while (infile.get() == '%') getline(infile, line);
    infile >> num_vertices >> num_edges;

    num_edges*=2; //because they are undericted

    int *v_adj_begin = new int[num_vertices];
    int *v_adj_length = new int[num_vertices];
    int *v_adj_list = new int[num_edges];

    if (verbose) printf("%s vertices=%d edges=%d\n", filename, num_vertices, num_edges);

    //int max_degree = 0;
    from = 0;

    getline(infile, line);
    while (getline(infile, line))
    {
        v_adj_begin[from] = edge_index;
        int degree = 0;

        //cout << line << endl;
        istringstream linestream(line); // #include <sstream>
        while (linestream >> to)
        {
            v_adj_list[edge_index] = to-1;
            edge_index++;
            degree++;
        }

        v_adj_length[from] = degree;

        from++;
    }

    // Generate data structure
    // printf("Build ajacency lists\n");

    for (int i = 0; i < 200; i++) {
        printf("%d ", v_adj_list[i]);
    }
    printf("\n");

    for (int i = 0; i < 60; i++) {
        printf("%d|", v_adj_begin[i]);
        printf("%d ", v_adj_length[i]);
    }
    printf("\n");

    graph_t *graph = new graph_t;
    graph->v_adj_list = v_adj_list;
    graph->v_adj_begin = v_adj_begin;
    graph->v_adj_length = v_adj_length;
    graph->num_vertices = num_vertices;
    graph->num_edges = num_edges;   
    return graph;
}


graph_t* load_graph()
{

    // Find number of vertices
    // printf("Reading input file\n");
    ifstream infile(filename);
    
    if (!infile.is_open()) 
    { 
        printf("File did not open\n");
        exit(1); 
    } else {
        printf("File opened\n");
    }

    //filename = basename(filename);
    
    int from, to;
    int num_edges = 0;
    int num_vertices = 0;

    map<int, int> index_map;
    int next_index = 0;
    string line;
    //skip the comments, num vertices and edges line
    while (infile.get() == '%') getline(infile, line);
    infile >> num_vertices >> num_vertices >> num_edges;

    int *v_adj_begin = new int[num_vertices];
    int *v_adj_length = new int[num_vertices];
    vector<int> *v_adj_lists = new vector<int>[num_vertices]();
    int *v_adj_list = new int[num_edges];

    if (verbose) printf("%s vertices=%d edges=%d\n", filename, num_vertices, num_edges);

    int max_degree = 0;

    while (infile >> from >> to)
    {
        if (!index_map.count(from))
        {
            index_map[from] = next_index++;
        }

        if (!index_map.count(to))
        {
            index_map[to] = next_index++;
        }

        v_adj_lists[index_map[from]].push_back(index_map[to]);
        max_degree = max(max_degree, (int) v_adj_lists[index_map[from]].size());
    }

    //num_vertices = next_index;
    //if (verbose) printf("vertices_read=%d\n", num_vertices);
    
    // Show degree distribution
    // printf("Compute out-degree histogram\n");
    int *degree_histogram = new int[max_degree + 1]();
    unsigned long long total_degree = 0;

    for (int i = 0; i < num_vertices; i++)
    {
        degree_histogram[v_adj_lists[i].size()]++;
        total_degree += v_adj_lists[i].size();
    }

    double avg_degree = total_degree / (double) num_vertices;
    double degree_variance = 0.0;

    for (int i = 0; i < num_vertices; i++)
    {
        degree_variance += (avg_degree - v_adj_lists[i].size()) * (avg_degree - v_adj_lists[i].size());
    }
    degree_variance /= num_vertices;

    double degree_stddev = sqrt(degree_variance);

    // Compute median
    int *degs = new int[num_vertices];

    for (int i = 0; i < num_vertices; i++)
    {
        degs[i] = v_adj_lists[i].size();
    }

    //sort(degs, degs + num_vertices);

    if (verbose) printf("avg_deg=%f deg_stddev=%f median=%i\n", avg_degree, degree_stddev, degs[num_vertices / 2]);
    
    /*
    printf("Histogram for Vertex Degrees\n");

    for (int i = 0; i < max_degree + 1; i++)
    {
        printf("deg %i        %i\n", i, degree_histogram[i]);
    }
    */

    // Generate data structure
    // printf("Build ajacency lists\n");
    int next_offset = 0;

    for (int i = 0; i < num_vertices; i++)
    {
        int list_size = v_adj_lists[i].size();
        
        // printf("\nvertex %d | begin = %d | size = %d :", i, next_offset, list_size);
        // for (int j = 0; j < list_size; j++)
        // {    
        //     printf(" %d", v_adj_lists[i][j]);
        // }

        v_adj_begin[i] = next_offset;
        v_adj_length[i] = list_size;

        memcpy(v_adj_list + next_offset, &v_adj_lists[i][0], list_size * sizeof(int));
        next_offset += list_size;
    }

    graph_t *graph = new graph_t;
    graph->v_adj_list = v_adj_list;
    graph->v_adj_begin = v_adj_begin;
    graph->v_adj_length = v_adj_length;
    graph->num_vertices = num_vertices;
    graph->num_edges = num_edges;   
    return graph;
}


graph_t* load_graph_double()
{

    // Find number of vertices
    // printf("Reading input file\n");
    ifstream infile(filename);
    
    if (!infile.is_open()) 
    { 
        printf("File did not open\n");
        exit(1); 
    }

    filename = basename(filename);
    
    int from, to;
    int num_edges = 0;
    int num_vertices = 0;

    map<int, int> index_map;
    int next_index = 0;
    string line;
    //skip the comments, num vertices and edges line
    while (infile.get() == '%') getline(infile, line);
    infile >> num_vertices >> num_vertices >> num_edges;
    num_vertices = num_vertices*2 + 1;
    num_edges = num_edges*2 + 1;

    int *v_adj_begin = new int[num_vertices];
    int *v_adj_length = new int[num_vertices];
    vector<int> *v_adj_lists = new vector<int>[num_vertices]();
    int *v_adj_list = new int[num_edges];

    if (verbose) printf("%s vertices=%d edges=%d\n", filename, num_vertices, num_edges);

    int max_degree = 0;

    while (infile >> from >> to)
    {
        if (!index_map.count(from))
        {
            index_map[from] = next_index++;
        }

        if (!index_map.count(to))
        {
            index_map[to] = next_index++;
        }

        v_adj_lists[index_map[from]].push_back(index_map[to]);
        max_degree = max(max_degree, (int) v_adj_lists[index_map[from]].size());
    }
    int vertices_read = next_index;
    if (verbose) printf("vertices_read=%d\n", next_index);

    v_adj_lists[vertices_read].push_back(0);
    v_adj_lists[vertices_read].push_back(vertices_read+1);

    for (int i = 0; i < vertices_read; i++)
    {
        int list_size = v_adj_lists[i].size();
        for (int j = 0; j < list_size; j++)
        {
            v_adj_lists[vertices_read+i+1].push_back(v_adj_lists[i][j]+vertices_read+1);
        }
    }

    int rand_vert = 232;
    int rand_list_size = v_adj_lists[rand_vert].size();

    for (int j = 0; j < rand_list_size; j++)
    {
        if (verbose) printf("%d ", v_adj_lists[rand_vert][j]);
    }

    printf("\n");

    for (int j = 0; j < rand_list_size; j++)
    {
        if (verbose) printf("%d ", v_adj_lists[rand_vert+vertices_read+1][j]);
    }
    
    // Show degree distribution
    // printf("Compute out-degree histogram\n");
    int *degree_histogram = new int[max_degree + 1]();
    unsigned long long total_degree = 0;

    for (int i = 0; i < num_vertices; i++)
    {
        degree_histogram[v_adj_lists[i].size()]++;
        total_degree += v_adj_lists[i].size();
    }

    double avg_degree = total_degree / (double) num_vertices;
    double degree_variance = 0.0;

    for (int i = 0; i < num_vertices; i++)
    {
        degree_variance += (avg_degree - v_adj_lists[i].size()) * (avg_degree - v_adj_lists[i].size());
    }
    degree_variance /= num_vertices;

    double degree_stddev = sqrt(degree_variance);

    // Compute median
    int *degs = new int[num_vertices];

    for (int i = 0; i < num_vertices; i++)
    {
        degs[i] = v_adj_lists[i].size();
    }

    //sort(degs, degs + num_vertices);

    if (verbose) printf("avg_deg=%f deg_stddev=%f median=%i\n", avg_degree, degree_stddev, degs[num_vertices / 2]);
    
    /*
    printf("Histogram for Vertex Degrees\n");

    for (int i = 0; i < max_degree + 1; i++)
    {
        printf("deg %i        %i\n", i, degree_histogram[i]);
    }
    */

    // Generate data structure
    // printf("Build ajacency lists\n");
    int next_offset = 0;

    for (int i = 0; i < num_vertices; i++)
    {
        int list_size = v_adj_lists[i].size();
        
        /*printf("\nvertex %d | begin = %d | size = %d :", i, next_offset, list_size);
        for (int j = 0; j < list_size; j++)
        {    
            printf(" %d", v_adj_lists[i][j]);
        }*/

        v_adj_begin[i] = next_offset;
        v_adj_length[i] = list_size;

        memcpy(v_adj_list + next_offset, &v_adj_lists[i][0], list_size * sizeof(int));
        next_offset += list_size;
    }

    graph_t *graph = new graph_t;
    graph->v_adj_list = v_adj_list;
    graph->v_adj_begin = v_adj_begin;
    graph->v_adj_length = v_adj_length;
    graph->num_vertices = vertices_read*2+1;
    graph->num_edges = num_edges;   
    return graph;
}


int main(int argc, char *argv[])
{
    if (argc < 5)
    {
        printf("Usage: %s -f filename(str) -n num_gpu(int) -b blocks(int) -t threads(int)\n", argv[0]);
        printf("Optional: -s start_vertex_id(int) -r runs(int) -m method(int) -l fma_load(int) ");
        printf("-o offload_threshold(float) -c communication_threshold(int) -v verbose(bool)\n");
        printf("-p use_metis_partitioning(bool)\n");
        exit(1);
    }

    filename = argv[1];
    int start_vertex = 0;
    runs = 1;
    int METHOD = -2;
    int opt;

    while ((opt = getopt(argc, argv, "f:b:t:s:r:n:m:l:o:c:vp")) != -1) {
        switch (opt) 
        {
            case 'f': filename = optarg; break;
            case 'b': BLOCKS = atoi(optarg); break;
            case 't': THREADS = atoi(optarg); break;
            case 's': start_vertex = atoi(optarg); break;
            case 'r': runs = atoi(optarg); break;
            case 'n': N_GPU = atoi(optarg); break;
            case 'm': METHOD = atoi(optarg); break;
            case 'o': OFF_THRESH = atof(optarg); break;
            case 'c': COMM_THRESH = atof(optarg); break;
            case 'v': verbose = true; break;
            case 'p': metis = true; break;
            case '?':
                if (isprint (optopt))
                    printf("Unknown option `-%c'.\n", optopt);
                else
                    printf("Option -%c requires an argument.\n", optopt); 
                exit(1);      
            default:
                exit(1);
        }
    }

    if (verbose)
    {
        printf("filename=%s, ", filename);
        printf("num_gpu=%d, ", N_GPU);
        printf("num_blocks=%d, ", BLOCKS);
        printf("num_threads=%d, ", THREADS);
        printf("num_runs=%d, ", runs);
        if (start_vertex < 0)
            printf("start_vertex=%s, ", start_vertex==-1 ? "ALL" : "SOME");
        else
            printf("start_vertex=%d, ", start_vertex);
        if (OFF_THRESH > 0) 
            printf("offload_threshold=%.2f*(num_blocks*num_threads), ", OFF_THRESH);
        else
            printf("no offloading, ");
        if (COMM_THRESH >= 0) 
            printf("comm_threshold=%d, ", COMM_THRESH);
        else
            printf("comm every iter, ");
        if (METHOD < 0)
            printf("METHOD=%s, ", METHOD==-1 ? "CPU" : "ALL");
        else
            printf("METHOD=%s, ", bfs_names[METHOD].c_str());
        if (metis)
            printf("metis partitioning\n");
        else
            printf("random partitioning\n");
    }

    struct timeval t1, t2;
    long long time;

    if (N_GPU == 0) cudaGetDeviceCount(&N_GPU);

    gettimeofday(&t1, NULL);

    graph_t* graph = load_graph();
    // int* v_owner;
    // if (metis) {
    //     v_owner = read_part(graph); 
    // }
    /*graph_t* metis_graph;
    if (metis) {
        metis_graph = metis_part(graph); 
    }*/

    //gettimeofday(&t2, NULL);
    //time = get_elapsed_time(&t1, &t2);
    //if (verbose) printf("Load=%lld\n", time);

    if (METHOD == -2) 
    {
        run_all_bfs(graph, start_vertex);
    } 
    else 
    {
        int range_from, range_to;
        int stride = 1;
        if (start_vertex == -1 || graph->num_vertices < RT_STUDY_NUM_V)
        {
            // Run for all start vertices
            range_from = 0;
            range_to = graph->num_vertices;
        }
        else if (start_vertex == -2)
        {
            // Run for RT_STUDY_NUM_V many vertices
            range_from = 0;
            range_to = graph->num_vertices;
            stride = graph->num_vertices / RT_STUDY_NUM_V;
        }
        else
        {
            range_from = start_vertex;
            range_to = start_vertex + 1;
        }

        for (int vertex = range_from; vertex < range_to; vertex+=stride)
        {
            if (verbose) printf("vertex=%d\n", vertex);

            int *expected = new int[graph->num_vertices];
            gettimeofday(&t1, NULL);

            bfs_sequential(graph, vertex, expected);

            gettimeofday(&t2, NULL);
            time = get_elapsed_time(&t1, &t2);
            if (verbose) printf("CPU=%lld\n", time);

            // if (metis) {
            //     int *result = new int[graph->num_vertices];

            //     int next_runtime = 1073741824;

            //     for (int i = 0; i < runs; i++)
            //     {
            //         // Reset result array
            //         memset(result, 0, sizeof(int) * graph->num_vertices);

            //         int next_time = workq_ring_part(
            //             graph->v_adj_list, 
            //             graph->v_adj_begin, 
            //             graph->v_adj_length, 
            //             v_owner,
            //             graph->num_vertices, 
            //             graph->num_edges,
            //             start_vertex, 
            //             result);

            //         next_runtime = min(next_time, next_runtime);
                    
            //         if (!equal(result, result + graph->num_vertices, expected))
            //         {
            //             int errs = 0;
            //             for (int m = 0; m < graph->num_vertices; m++)
            //             {
            //                 if (expected[m] != result[m])
            //                 {
            //                     errs++;
            //                     if (errs < 10) printf("Vertex %d: %d vs %d\n", m, result[m], expected[m]);
            //                 }
            //             }
            //             printf("Completed with %d errors\n", errs);
            //             // Wrong result
            //             return -1;
            //         }
            //     }

            //     free(result);
            // }

            if (METHOD >= 0)
            {
                int next_runtime = run_bfs(bfs_functions[METHOD], graph, vertex, expected, runs);
                //printf("GPU_all=%d\n", std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count());
                //printf("%s,%s,%i,%i,%i,%d\n", filename, bfs_names[METHOD].c_str(), N_GPU, BLOCKS, THREADS, next_runtime);
            }
            delete[] expected;
        }
    }
    delete[] graph->v_adj_list;
    delete[] graph->v_adj_begin;
    delete[] graph->v_adj_length;
    //delete[] metis_graph->v_adj_list;
    //delete[] metis_graph->v_adj_begin;
    //delete[] metis_graph->v_adj_length;
    //delete[] metis_graph->split_offsets;
    delete graph;
    //delete metis_graph;
    // printf("\n");
    gettimeofday(&t2, NULL);
    time = get_elapsed_time(&t1, &t2);
    printf("elapsed time=%lld\n", time); 
}
