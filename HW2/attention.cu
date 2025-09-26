#include "common.cuh"
#include <cmath>

__global__ void transpose_LD_to_DL(const double *__restrict__ K, double *__restrict__ KT, int L, int D)
{
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < L && col < D)
    KT[col * L + row] = K[row * D + col];
}

__global__ void softmax_rows(double *__restrict__ S, int L, double inv_sqrt_d)
{
  // write your softmax code similar to HW1
  int row_index = blockIdx.y; // blockDim.y = 1
  double *row = S + row_index * L;
  double sum = 0.0;
  double m = row[0];

  for (int i = 0; i < L; i++)
    if (row[i] > m)
      m = row[i];

  for (int i = 0; i < L; i++)
  {
    row[i] = std::exp(row[i] - m);
    sum += row[i];
  }

  for (int i = 0; i < L; i++)
    row[i] = row[i] * inv_sqrt_d / sum;
}

void dgemm_naive_gpu(const double *, const double *, double *, int, int, int, cudaStream_t);
void dgemm_tiled_gpu(const double *, const double *, double *, int, int, int, int, cudaStream_t);

void attention_via_dgemm(const double *dQ, const double *dK, const double *dV, double *dO, int L, int D, int mode, cublasHandle_t h, cudaStream_t stream = 0)
{
  // Write your code here
  // You can use mode to call different versions of DGEMM
  // For example
  double *dKT = nullptr;
  double *dS = nullptr;
  CUDA_CHECK(cudaMalloc(&dKT, L * D * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&dS, L * L * sizeof(double)));

  int block_dim = 16;
  dim3 block(block_dim, block_dim);
  int rows = (L + block.y - 1) / block.y;
  int cols = (L + block.x - 1) / block.x;
  dim3 grid(cols, rows);

  transpose_LD_to_DL<<<grid, block, 0, stream>>>(dK, dKT, L, D);

  if (mode == 0)
    dgemm_naive_gpu(dQ, dKT, dS, L, L, D, stream);
  else if (mode == 1)
    dgemm_tiled_gpu(dQ, dKT, dS, L, L, D, 32, stream);

  double inv_sqrt_d = 1.0 / std::sqrt((double)D);
  dim3 softmax_block(1, 1);
  dim3 softmax_grid(1, L);

  softmax_rows<<<softmax_grid, softmax_block, 0, stream>>>(dS, L, inv_sqrt_d);

  if (mode == 0)
    dgemm_naive_gpu(dS, dV, dO, L, D, L, stream);
  else if (mode == 1)
    dgemm_tiled_gpu(dS, dV, dO, L, D, L, 32, stream);
}
