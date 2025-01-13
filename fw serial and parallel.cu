#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <limits.h>

#define INFNTY INT_MAX

int *adjacency_matrix, *dp_matrix;

/* Undirected graph non-negative edge weights */
/* This function generates linearized 2D array of length N*N */
void generate_random_adj_matrix(int n_vertices)
{
    int N = n_vertices;
    int i, j;

    /* Allocate memory for adjacency matrix 2D array */
    adjacency_matrix = malloc(N * N * sizeof(int));


    srand(0);
    for (i = 0; i < N; i++)
    {
        for (j = i; j < N; j++)
        {
            if (i == j)
            {
                adjacency_matrix[i*N+j] = 0; /* Diagonal */
                adjacency_matrix[j*N+i] = 0; /* Diagonal */
            }
            else
            {
                /* Zero to nine random */
                int r = rand() % 10;
                int val = (r == 2) ? INFNTY : r; /* No edge between vertices */
                adjacency_matrix[i*N + j] = val;    /* Symmetrically */
                adjacency_matrix[j*N +i] = val;
            }
        }
    }
}
_global_ void generate_random_adj_matrix_kernel(int N){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  srand(0);
  if (i == j){
    adjacency_matrix[i*N+j] = 0;
    adjacency_matrix[j*N+i] = 0;
  }
  else{
    int r = rand() % 10;
    int val = (r == 2) ? INFNTY : r; /* No edge between vertices */
    adjacency_matrix[i*N + j] = val;
    adjacency_matrix[j*N + i] = val;
  }
}

void floyd_warshall_serial(int **graph, int **dp, int N)
{
    int i, j, k;
    /* Initialize copy graph to dp matrix */
    for (i = 0; i < N; i++)
        for (j = 0; j < N; j++)
            dp[i][j] = graph[i][j];

    /* Floyd Warshall algorithm */
    for (k = 0; k < N; k++)
    {
        for (i = 0; i < N; i++)
        {
            for (j = 0; j < N; j++)
            {
                if (dp[i][k] + dp[k][j] < dp[i][j])
                    dp[i][j] = dp[i][k] + dp[k][j];
            }
        }
    }
}
// 10000
_global_ void floyd_warshall_kernel(int *dp, int N, int k) {
  int i, j;
  i = blockIdx.x * blockDim.x + threadIdx.x;
  j = blockIdx.y * blockDim.y + threadIdx.y;

  if (i < N && j < N) {
    if(dp[i*N+k] != INFNTY && dp[k*N+j] != INFNTY) {
      if (dp[i*N+k]+ dp[k*N+j] < dp[i*N+j]) {
        dp[i*N+j] = dp[i*N+k] + dp[k*N+j];
      }
    }
  }
}



int main(int argc, char **argv)
{
    if (argc != 2)
    {
        printf("USAGE: ./floyd_serial <number_of_vertices>\n");
        return 1;
    }

    int n_vertices;
    n_vertices = atoi(argv[1]);

    // Allocate memory for matrices
    dp_matrix = malloc(n_vertices * n_vertices * sizeof(int));

    generate_random_adj_matrix(n_vertices);

    clock_t start = clock(); /* Start measuring execution time */



    // Define the number of streams.
    const uint64_t num_streams = 32;
    const int blocksPerGrid = 32;
    const int threadsPerBlock = 1024;

    const uint64_t chunk_size = sdiv(n_vertices, num_streams);
    uint64_t * data_cpu, * data_gpu;
    cudaMallocHost(&data_cpu, sizeof(uint64_t)n_vertices n_vertices);
    cudaMalloc    (&data_gpu, sizeof(uint64_t)n_vertices n_vertices);
    check_last_error();

    cudaStream_t streams[num_streams];

    for (uint64_t i = 0; i < num_streams; i++){
      cudaStreamCreate(&streams[i]);
    }
    check_last_error();



    for (uint64_t stream = 0; stream < num_streams; stream++) {

        // ...calculate index into global data (lower) and size of data for it to process (width).
        const uint64_t lower = chunk_size*stream;
        const uint64_t upper = min(lower+chunk_size, n_vertices);
        const uint64_t width = upper-lower;

        // ...copy stream's chunk to device.
        cudaMemcpyAsync(data_gpu+lower, data_cpu+lower,
               sizeof(uint64_t)*width, cudaMemcpyHostToDevice,
               streams[stream]);
        floyd_warshall_kernel<<<blocksPerGrid, threadsPerBlock, 0, streams[stream]>>>();

//        // ...compute stream's chunk.
//        decrypt_gpu<<<80*32, 64, 0, streams[stream]>>>
//            (data_gpu+lower, width, num_iters);
//
//        // ...copy stream's chunk to host.
//        cudaMemcpyAsync(data_cpu+lower, data_gpu+lower,
//               sizeof(uint64_t)*width, cudaMemcpyDeviceToHost,
//               streams[stream]);
    }


    for (uint64_t stream = 0; stream < num_streams; stream++){
        cudaStreamSynchronize(streams[stream]);
    }

    timer.stop("total time on GPU");
    check_last_error();


    free(adjacency_matrix);
    free(dp_matrix);

    return 0;
}