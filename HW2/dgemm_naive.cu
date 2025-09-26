#include "common.cuh"
#include <cuda_runtime.h>

// Device code
__global__ void dgemm_naive_kernel(const double *__restrict__ A,
                                   const double *__restrict__ B,
                                   double *__restrict__ C, int M, int N,
                                   int K)
{
  // C (M x N) = A (M x K) * B (K x N)
  int i = blockIdx.y * blockDim.y + threadIdx.y; // ith row of C
  int j = blockIdx.x * blockDim.x + threadIdx.x; // jth column of C

  // Guard against out-of-bounds threads when M or N are not multiples of blockDim
  if (i >= M || j >= N)
    return;

  double sum = 0.0;
  for (int k = 0; k < K; k++)
  {
    sum += A[i * K + k] * B[k * N + j];
  }

  C[i * N + j] = sum;
}

// Host code
void dgemm_naive_gpu(const double *dA, const double *dB, double *dC, int M,
                     int N, int K, cudaStream_t stream = 0)
{
  // C (M x N) = A (M x K) * B (K x N)
  int block_dim = 16;
  dim3 block(block_dim, block_dim);
  int cols = (N + block.x - 1) / block.x;
  int rows = (M + block.y - 1) / block.y;
  dim3 grid(cols, rows);

  dgemm_naive_kernel<<<grid, block, 0, stream>>>(dA, dB, dC, M, N, K);
}
