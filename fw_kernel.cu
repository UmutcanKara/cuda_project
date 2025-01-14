
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <limits.h>
#include <curand_kernel.h>
#define INFNTY INT_MAX
#include "helpers.cuh"
#include "encryption.cuh"




__global__ void setup_kernel(curandState *state, unsigned long seed) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(seed, id, 0, &state[id]); // Initialize state
}


/* Undirected graph non-negative edge weights */
/* This function generates linearized 2D array of length N*N */

__global__ void generate_random_adj_matrix_kernel(int N, int *adjacency_matrix ,curandState *state, int *randomNumbers) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;


  if (i == j){
    adjacency_matrix[i*N+j] = 0;
    adjacency_matrix[j*N+i] = 0;
  }
  else{
    int r;
    float rand = curand_uniform(state);
    curandState localState = state[i];
    r = (int)(curand_uniform(&localState) * 10.0f);


    int val = (r == 2) ? INFNTY : r; /* No edge between vertices */
    adjacency_matrix[i*N + j] = val;
    adjacency_matrix[j*N + i] = val;
  }
}


__global__ void floyd_warshall_kernel(int *dp, int N, int k) {
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

__global__ void print_adjacency_matrix_kernel(int V, int *adjacency_matrix)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    for(int i=tid; i < V; i+=stride)
        {
            printf("%d\n",adjacency_matrix[i]);
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

    dim3 blocksPerGrid = (16,16);
    dim3 threadsPerBlock = (32,32);

    int *adjacency_matrix, *dp_matrix;




    // Allocate memory for matrices
    cudaMalloc(&dp_matrix, n_vertices * n_vertices * sizeof(int));
    cudaMalloc(&adjacency_matrix, n_vertices * n_vertices * sizeof(int));

    // init curandom
    curandState *d_states;
    int* d_randomNumbers;
    cudaMalloc(&d_randomNumbers, n_vertices * n_vertices * sizeof(int));
    cudaMalloc(&d_states, n_vertices * n_vertices*  sizeof(curandState));

    // Setup CURAND states
    setup_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_states, time(NULL));

    clock_t start1 = clock(); /* Start measuring execution time */
    generate_random_adj_matrix_kernel<<<blocksPerGrid,threadsPerBlock>>>(n_vertices,adjacency_matrix,  d_states, d_randomNumbers);
    cudaDeviceSynchronize();

    clock_t end1 = clock();
    clock_t start2 = clock(); /* Start measuring execution time */
    for (int i = 0; i < n_vertices; i++){
      floyd_warshall_kernel<<<blocksPerGrid, threadsPerBlock>>>(adjacency_matrix, n_vertices, i);
      cudaDeviceSynchronize();
    }
    check_last_error();
    clock_t end2 = clock();

    cudaFree(adjacency_matrix);
    cudaFree(dp_matrix);

    float seconds;
    seconds = (float)(end1 - start1) / CLOCKS_PER_SEC;
    printf("TIME FOR GRAPH GENERATION ON GPU = %f SECS\n", seconds);
    seconds = (float)(end2 - start2) / CLOCKS_PER_SEC;
    printf("TIME FOR ALL PAIRS Floyd ON GPU = %f SECS\n", seconds);



    return 0;
}
