#include <algorithm>
#include <cmath>
#include <omp.h>
#include <vector>

void dgemm_naive(const double *A, const double *B, double *C, int m, int n,
                 int k);
// Helper: transpose K(LxD) to KT(DxL)
static void transpose(const double *K, double *KT, int L, int D) {
#pragma omp parallel for collapse(2)
  for (int i = 0; i < L; i++) {
    for (int j = 0; j < D; j++) {
      KT[j * L + i] = K[i * D + j];
    }
  }
}

// Helper: row-wise softmax
static void softmax_rows(double *S, int L) {
#pragma omp parallel
  {
#pragma omp for
    for (int i = 0; i < L; i++) { // L * 60 * L
      double *row = S + i * L;
      double m = row[0];
      for (int j = 1; j < L; j++)
        m = std::max(m, row[j]); // L
      double sum = 0.0;
#pragma omp simd reduction(+ : sum)
      for (int j = 0; j < L; j++) {
        row[j] = std::exp(row[j] - m); // 51 * L
        sum += row[j];                 // L
      }
#pragma omp simd
      for (int j = 0; j < L; j++)
        row[j] /= sum; // 8 * L
    }
  }
}

void attention_via_dgemm(const double *Q, const double *K, const double *V,
                         double *O, int L, int D, bool use_blocked, int BM,
                         int BN, int BK) {

  const std::size_t Ksz = (std::size_t)D * L;
  const std::size_t Ssz = (std::size_t)L * L;
  std::vector<double> S(Ssz), KT(Ksz, 0.0);

  transpose(K, KT.data(), L, D);
  dgemm_naive(Q, KT.data(), S.data(), L, L, D); // 2 * L * L * D
  double scale = 1.0 / std::sqrt((double)D);

#pragma omp parallel for
  for (int i = 0; i < L; i++) { // L * L
#pragma omp simd
    for (int j = 0; j < L; j++) {
      S[i * L + j] = scale * S[i * L + j];
    }
  }

  softmax_rows(S.data(), L);            // 60 * L  * L
  dgemm_naive(S.data(), V, O, L, D, L); // 2 * L * D * L
                                        // Total flops = 4LLD + 60LL
}
