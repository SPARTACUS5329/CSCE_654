#include <algorithm>
#include <cmath>
#include <memory>
#include <omp.h>
#include <vector>

// Direct loop-based attention (no DGEMM)
// We added suggested template without OpenMP
// Feel free to modify

void attention_direct(const double *Q, const double *K, const double *V,
                      double *O, int L, int D) {
  double scale = 1.0 / std::sqrt((double)D);

#pragma omp parallel for
  for (int i = 0; i < L; i++) {
#pragma omp simd
    for (int j = 0; j < D; j++) {
      O[i * D + j] = 0.0;
    }
  }

  // Initialize the output matrix
  const std::size_t Ssz = (std::size_t)L * L;
  std::vector<double> S(Ssz);

  for (int i = 0; i < L; i++) {
    // compute one row of S through a dot product
#pragma omp parallel for
    for (int j = 0; j < L; j++) {
      double sum = 0.0;
#pragma omp simd reduction(+ : sum)
      for (int d = 0; d < D; d++)
        sum += Q[i * D + d] * K[j * D + d];
      S[i * L + j] = scale * sum;
    }

    // Compute softmax for this row

    double *row = S.data() + i * L;
    double m = row[0];
    for (int j = 1; j < L; j++)
      m = std::max(m, row[j]);
    double sum = 0.0;
#pragma omp simd reduction(+ : sum)
    for (int j = 0; j < L; j++) {
      row[j] = std::exp(row[j] - m);
      sum += row[j];
    }
#pragma omp simd
    for (int j = 0; j < L; j++)
      row[j] /= sum;

    // ith row of S now equal to ith row of A
    // O[i,:] = sum_j(A[i, j] * V[j,:]) -> 0 <= j < L
    // O[i, j] = sum(A[i,:] * V[:,j]) -> 0 <= j < L

#pragma omp parallel for
    for (int d = 0; d < D; d++) {
      double sum = 0.0;
#pragma omp simd reduction(+ : sum)
      for (int j = 0; j < L; j++) {
        sum += S[i * L + j] * V[j * D + d];
      }
      O[i * D + d] = sum;
    }
  }
}
