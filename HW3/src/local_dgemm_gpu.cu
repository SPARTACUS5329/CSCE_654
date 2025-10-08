#include "localmatrix.hpp"
#include "common.hpp"
#ifdef ENABLE_CUDA
#include <cuda_runtime.h>

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
        // sum += A(i, k) * B(k, j) and B is col-major
        sum += A[i * K + k] * B[j * N + k];
    }

    C[i * N + j] += sum; // Because sub-sums are calculated
}

void local_dgemm_gpu(double *dC,
                     const double *dA, int Am, int An,
                     const double *dB, int Bm, int Bn,
                     cudaStream_t stream = 0)
{
    // C (Am x Bn)
    dim3 block(4, 2);
    int cols = (Bn + block.x - 1) / block.x;
    int rows = (Am + block.y - 1) / block.y;
    dim3 grid(cols, rows);

    dgemm_naive_kernel<<<grid, block, 0, stream>>>(dA, dB, dC, Am, Bn, An);
}
#endif
