#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <limits.h>
#include <curand_kernel.h>

#define TRUE 1
#define FALSE 0
#define INFNTY INT_MAX

typedef int boolean;

#include <curand_kernel.h>

__global__ void setup_kernel(curandState *state, unsigned long seed) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(seed, id, 0, &state[id]); // Initialize state
}


/* Generates a random undirected graph represented by an adjacency matrix */
__global__ void generate_random_graph_kernel(int V, int *adjacency_matrix, curandState *state, int *randomNumbers)
{    
    int tid = blockIdx.x*blockDim.x+threadIdx.x;
    int stride = blockDim.x*gridDim.x;

    for (int i = tid; i < V; i+=stride)
    {
        for (int j = 0; j < V; j++)
        {
            int r;
            curandState localState = state[i];
            r = (int)(curand_uniform(&localState) * 10.0f);

            if (i < j) {
                adjacency_matrix[i * V + j] = r;                 /* Assign a random value corresponding to the edge */
                adjacency_matrix[j * V + i] = r; /* Graph is undirected, the adjacency matrix is symmetric */
               }
            state[i] = localState;
            }
        adjacency_matrix[i * V + i] = 0;
        
        }
    }


__global__ void print_adjacency_matrix_kernel(int V, int *adjacency_matrix)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    for(int i=tid; i < V; i+=stride)
        {
            printf("matrix[%d] = %d\n",i,adjacency_matrix[i]);
        }
}


__global__ void dijkstra_kernel(int V, int *adjacency_matrix, int *len, int *temp_distance, boolean *visited)
{

    /* Computing the All Pairs Shortest Paths (APSP) in the graph */
    int tid = blockIdx.x*blockDim.x+threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(int source = tid; source < V; source+=stride)
        {
        for (int i = 0; i < V; i++) /* Initialize vars arrays to current source */
        {
            visited[i] = FALSE;
            temp_distance[i] = INFNTY;
            len[source * V + i] = INFNTY;
        }

        len[source * V + source] = 0; /* Set the distance of the source vertex as 0 */

        for (int count = 0; count < V - 1; count++)
        {
            /* Finds the vertex with the minimum distance from the current source vertex */
            int min_distance = INFNTY; /* Init value */
            int min_index = -1;
        
            for (int v = 0; v < V; v++) /* Iterates over all vertices */
            {
                if (!visited[v] && len[v] <= min_distance)
                {
                    min_distance = len[v];
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
}



int main(int argc, char **argv)
{

    if (argc != 2)
    {
        printf("USAGE: ./dijkstra_parallel <number_of_vertices>\n");
        return 1;
    }
    
    int deviceId;
    cudaGetDevice(&deviceId);
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, deviceId);
    int sm_count = props.multiProcessorCount;
    int warp_size = props.warpSize;
    
    int threadsPerBlock = warp_size*16;
    int numberOfBlocks = sm_count*16;

    int V = atoi(argv[1]); /* Number of vertices */
    int * adjacency_matrix;
    int *len, *temp_distance;
    boolean *visited;
    
    cudaMalloc(&visited, V * sizeof(boolean));
    cudaMalloc(&len, V * V * sizeof(int));
    cudaMalloc(&adjacency_matrix, V * V * sizeof(int));
    cudaMalloc(&temp_distance, V * sizeof(boolean));
    
    curandState *d_states;
    int* d_randomNumbers;
    cudaMalloc(&d_randomNumbers, V * V * sizeof(int));
    cudaMalloc(&d_states, V * V * sizeof(curandState));

    // Setup CURAND states
    setup_kernel<<<numberOfBlocks, threadsPerBlock>>>(d_states, time(NULL));
    
    clock_t start = clock(); /* Records the start time for measuring the execution time */
    generate_random_graph_kernel<<<numberOfBlocks, threadsPerBlock>>>(V, adjacency_matrix, d_states, d_randomNumbers);
    cudaDeviceSynchronize();
    clock_t end = clock();   /* Records the end time for measuring the execution time */
    float seconds = (float)(end - start) / CLOCKS_PER_SEC;
    printf("TIME TO CREATE GRAPH ON GPU = %f SECS\n", seconds);
    
    start = clock(); /* Records the start time for measuring the execution time */

    dijkstra_kernel<<<numberOfBlocks, threadsPerBlock>>>(V, adjacency_matrix, len, temp_distance, visited);
    cudaDeviceSynchronize();
    end = clock(); /* Records the end time for measuring the execution time */
    seconds = (float)(end - start) / CLOCKS_PER_SEC;
    printf("TIME FOR ALL PAIRS DIJKSTRA ON GPU = %f SECS\n", seconds);
    
    cudaFree(visited);
    cudaFree(len);
    cudaFree(temp_distance);
    cudaFree(adjacency_matrix);
    
    return 0;
}
