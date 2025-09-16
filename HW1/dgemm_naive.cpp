#include <omp.h>

int NUM_THREADS = 32;

// Naive DGEMM: C(m x n) = A(m x k) * B(k x n)
// Row-major, flattened arrays
void dgemm_naive(const double *A, const double *B, double *C, int m, int n,
                 int k) {
  // Zero C
  // parallelize with OpenMP
#pragma omp parallel for
  for (int i = 0; i < m; i++) {
#pragma omp simd
    for (int j = 0; j < n; j++) {
      C[i * n + j] = 0.0;
    }
  }

  // TODO: Add OpenMP parallelization
  // Write your code here

#pragma omp parallel for schedule(dynamic, 128) num_threads(NUM_THREADS)
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      // C[i, j] = A[i,:] * B[:,j]
      // A[i,:] = A[i * k + 0] ... A[i * k + (k - 1)] -> k elements
      // B[:,j] = B[0 * n + j] ... B[(k - 1) * n + j] -> k elements
      // ith row of A and jth col of B
#pragma omp simd
      for (int cx = 0; cx < k; cx++) {
        C[i * n + j] += A[i * k + cx] * B[cx * n + j];
      }
    }
  }
}
