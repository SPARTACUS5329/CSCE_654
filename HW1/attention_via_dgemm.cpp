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
    for (int i = 0; i < L; i++) {
      double *row = S + i * L;
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
  dgemm_naive(Q, KT.data(), S.data(), L, L, D);
  double scale = 1.0 / std::sqrt((double)D);

#pragma omp parallel for
  for (int i = 0; i < L; i++) {
#pragma omp simd
    for (int j = 0; j < L; j++) {
      S[i * L + j] = scale * S[i * L + j];
    }
  }

  softmax_rows(S.data(), L);
  dgemm_naive(S.data(), V, O, L, D, L);
}
