#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <limits.h>

#define TRUE 1
#define FALSE 0
#define INFNTY INT_MAX

typedef int boolean;

/* Generates a random undirected graph represented by an adjacency matrix */
__global__ void generate_random_graph(int V, int *adjacency_matrix)
{
    srand(time(NULL));
    int tid = blockIdx.x*blockDim.x+threadIdx.x;
    int stride = blockDim.x*gridDim.x;

    for (int i = tid; i < V; i+=stride)
    {
        for (int j = 0; j < V; j++)
        {
            if (i != j)
            {
                adjacency_matrix[i * V + j] = rand() % 10;                 /* Assign a random value corresponding to the edge */
                adjacency_matrix[j * V + i] = adjacency_matrix[i * V + j]; /* Graph is undirected, the adjacency matrix is symmetric */
            }
            else
            {
                adjacency_matrix[i * V + j] = 0;
            }
        }
    }
}

/* Print adjacency matrix */
void print_adjacency_matrix(int V, int *adjacency_matrix)
{
    printf("\nADJACENCY MATRIX:\n");
    for (int i = 0; i < V; i++)
    {
        for (int j = 0; j < V; j++)
        {
            printf("%d ", adjacency_matrix[i * V + j]);
        }
        printf("\n");
    }
}

/* Finds vertex with the minimum distance among the vertices that have not been visited yet */
int find_min_distance(int V, int *distance, boolean *visited)
{
    int min_distance = INFNTY; /* Init value */
    int min_index = -1;

    for (int v = 0; v < V; v++) /* Iterates over all vertices */
    {
        if (!visited[v] && distance[v] <= min_distance)
        {
            min_distance = distance[v];
            min_index = v;
        }
    }
    return min_index;
}

__global__ void dijkstra_kernel(int V, int *adjacency_matrix, int *len, int *temp_distance, boolean *visited)
{

    clock_t start = clock(); /* Records the start time for measuring the execution time */

    /* Computing the All Pairs Shortest Paths (APSP) in the graph */
    int tid = blockIdx.x*blockDim.x+threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(int source = tid; source < V; source+=stride)
        {
        init_vars<<<40,32>>>(len, temp_distance, visited, source);

        len[source * V + source] = 0; /* Set the distance of the source vertex as 0 */

        for (int count = 0; count < V - 1; count++)
        {
            /* Finds the vertex with the minimum distance from the current source vertex */
            int min_distance = INFNTY; /* Init value */
            int min_index = -1;
        
            for (int v = 0; v < V; v++) /* Iterates over all vertices */
            {
                if (!visited[v] && distance[v] <= min_distance)
                {
                    min_distance = distance[v];
                    min_index = v;
                }
            }
            
            int current_vertex = min_index;
            visited[current_vertex] = TRUE;

            for (int v = 0; v < V; v++)
            {
                int weight = adjacency_matrix[current_vertex * V + v];
                if (!visited[v] && weight && len[source * V + current_vertex] != INFNTY &&
                    len[source * V + current_vertex] + weight < len[source * V + v])
                {
                    /* Updating the distance is beneficial */
                    len[source * V + v] = len[source * V + current_vertex] + weight;
                    temp_distance[v] = len[source * V + v];
                }
            }
        }
        }

    /* Records the end time for measuring the execution time */
    clock_t end = clock();
    float seconds = (float)(end - start) / CLOCKS_PER_SEC;
    printf("TOTAL ELAPSED TIME ON GPU = %f SECS\n", seconds);
}

__global__ void init_vars(int *len, int *temp_distance, boolean *visited, int source)
{
    int tid = blockIdx.x*blockDim.x+threadIdx.x;
    int stride = blockDim.x * gridDim.x;
        for (int i = tid; i < V; i+=stride) /* Initialize vars arrays to current source */
        {
            visited[i] = FALSE;
            temp_distance[i] = INFNTY;
            len[source * V + i] = INFNTY;
        }
}

__global__ void dijkstra_kernel_single_source(int V, int *adjacency_matrix, int *len, int *temp_distance, boolean *visited)
{

    clock_t start = clock(); /* Records the start time for measuring the execution time */

    /* Computing the Single Source Shortest Path in the graph */
    int tid = blockIdx.x*blockDim.x+threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int source = 0;
    
    init_vars<<<40,32>>>(len, temp_distance, visited, source);
    
    len[source * V + source] = 0; /* Set the distance of the source vertex as 0 */

        for (int count = 0; count < V - 1; count++)
        {
            /* Finds the vertex with the minimum distance from the current source vertex */
            int min_distance = INFNTY; /* Init value */
            int min_index = -1;
        
            for (int v = 0; v < V; v++) /* Iterates over all vertices */
            {
                if (!visited[v] && distance[v] <= min_distance)
                {
                    min_distance = distance[v];
                    min_index = v;
                }
            }
            
            int current_vertex = min_index;
            visited[current_vertex] = TRUE;

            for (int v = 0; v < V; v++)
            {
                int weight = adjacency_matrix[current_vertex * V + v];
                if (!visited[v] && weight && len[source * V + current_vertex] != INFNTY &&
                    len[source * V + current_vertex] + weight < len[source * V + v])
                {
                    /* Updating the distance is beneficial */
                    len[source * V + v] = len[source * V + current_vertex] + weight;
                    temp_distance[v] = len[source * V + v];
                }
            }
        }

    /* Records the end time for measuring the execution time */
    clock_t end = clock();
    float seconds = (float)(end - start) / CLOCKS_PER_SEC;
    printf("TOTAL ELAPSED TIME ON GPU = %f SECS\n", seconds);
}

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        printf("USAGE: ./dijkstra_serial <number_of_vertices>\n");
        return 1;
    }
    
    int num_gpus;
    // how many gpus
    cudaGetDeviceCount(&num_gpus);
    // properties for each gpu
    cudaDeviceProperties * props [num_gpus];
    cudaMallocHost(&props, num_gpus * sizeof(cudaDeviceProperties));
    for (int gpu = 0; gpu < num_gpus; gpu++)
        {
            cudaSetDevice(gpu);
            props[gpu] = cudaGetDeviceProperties(gpu);
        }
    
    int V = atoi(argv[1]); /* Number of vertices */
    int *len, *temp_distance;
    boolean *visited;
    cudaMalloc(&visited, V * sizeof(boolean));
    int num_streams = 10; // streams per gpu

    int len_gpu_chunk_size = (V*V+num_gpus-1)/num_gpus;

    
    // Each stream needs num_entries/num_gpus/num_streams data. We use round up division for
    // reasons previously discussed.
    int len_stream_chunk_size = (len_gpu_chunk_size+num_streams-1)/num_streams;
    

    // 2D array containing number of streams for each GPU.
    cudaStream_t streams[num_gpus][num_streams];

    // For each available GPU device...
    for (uint64_t gpu = 0; gpu < num_gpus; gpu++) {
        // ...set as active device...
        cudaSetDevice(gpu);
        for (uint64_t stream = 0; stream < num_streams; stream++)
            // ...create and store its number of streams.
            cudaStreamCreate(&streams[gpu][stream]);
    }
    check_last_error();

    
    
    // For each gpu device...
    for (uint64_t gpu = 0; gpu < num_gpus; gpu++) {

        // ...set device as active...
        cudaSetDevice(gpu);
        cudaMalloc(&len, V * V * sizeof(int));
        // ...use a GPU chunk's worth of data to calculate indices and width...
        const uint64_t lower = gpu_chunk_size*gpu;
        const uint64_t upper = min(lower+gpu_chunk_size, num_entries);
        const uint64_t width = upper-lower;

        // ...allocate data.
        cudaMalloc(&data_gpu[gpu], sizeof(uint64_t)*width);
    }

    
    temp_distance = (int *)malloc(V * sizeof(int));

    int *adjacency_matrix = (int *)malloc(V * V * sizeof(int));

    generate_random_graph(V, adjacency_matrix);
    dijkstra_serial(V, adjacency_matrix, len, temp_distance);

    /* print_adjacency_matrix(V, adjacency_matrix); */
    cudaFree(visited);
    cudaFree(len);
    cudaFree(temp_distance);
    cudaFree(adjacency_matrix);
    
    return 0;
}
