#include "common.cuh"

// I am using TILE as a C++ template.
// You can use  __shared__ double As[TILE][TILE] to allocate memory in this case.

template <int TILE>
__global__ void dgemm_tiled_kernel(const double *__restrict__ A,
                                   const double *__restrict__ B,
                                   double *__restrict__ C,
                                   int M, int N, int K)
{
  __shared__ double As[TILE][TILE];
  __shared__ double Bs[TILE][TILE];
  double sum = 0.0;
  int num_tiles = (K + TILE - 1) / TILE;

  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  for (int t = 0; t < num_tiles; t++)
  {
    int tiled_row = row;
    int tiled_col = t * TILE + threadIdx.x;

    if (tiled_row < M && tiled_col < K)
      As[threadIdx.y][threadIdx.x] = A[tiled_row * K + tiled_col];
    else
      As[threadIdx.y][threadIdx.x] = 0.0;

    tiled_row = t * TILE + threadIdx.y;
    tiled_col = col;
    if (tiled_row < K && tiled_col < N)
      Bs[threadIdx.y][threadIdx.x] = B[tiled_row * N + tiled_col];
    else
      Bs[threadIdx.y][threadIdx.x] = 0.0;

    __syncthreads();

    for (int k = 0; k < TILE; k++)
      sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];

    __syncthreads();
  }

  if (row < M && col < N)
    C[row * N + col] = sum;
}

// Host code
void dgemm_tiled_gpu(const double *dA, const double *dB, double *dC,
                     int M, int N, int K, int tile = 32, cudaStream_t stream = 0)
{
  // define grids and blocks and call the CUDA kernel
  int block_dim = tile;
  dim3 block(block_dim, block_dim);
  int rows = (M + block.y - 1) / block.y;
  int cols = (N + block.x - 1) / block.x;
  dim3 grid(cols, rows);
  switch (tile)
  {
  case 1:
    dgemm_tiled_kernel<1><<<grid, block, 0, stream>>>(dA, dB, dC, M, N, K);
    break;
  case 4:
    dgemm_tiled_kernel<4><<<grid, block, 0, stream>>>(dA, dB, dC, M, N, K);
    break;
  case 8:
    dgemm_tiled_kernel<8><<<grid, block, 0, stream>>>(dA, dB, dC, M, N, K);
    break;
  case 16:
    dgemm_tiled_kernel<16><<<grid, block, 0, stream>>>(dA, dB, dC, M, N, K);
    break;
  case 32:
    dgemm_tiled_kernel<32><<<grid, block, 0, stream>>>(dA, dB, dC, M, N, K);
    break;
  default:
    dgemm_tiled_kernel<16><<<grid, block, 0, stream>>>(dA, dB, dC, M, N, K);
    break;
  }
}
