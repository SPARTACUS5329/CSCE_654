#include "verify_mkl.hpp"
#include <cstdio>
#include <cstring>
#include <omp.h>
#include <random>
#include <vector>
extern int NUM_THREADS;

void dgemm_naive(const double *A, const double *B, double *C, int m, int n,
                 int k);
void attention_via_dgemm(const double *Q, const double *K, const double *V,
                         double *O, int L, int D, bool use_blocked, int BM,
                         int BN, int BK);
void attention_direct(const double *Q, const double *K, const double *V,
                      double *O, int L, int D);

static int argi(char **a, int i) { return std::atoi(a[i]); }

int main(int argc, char **argv) {
  int m = 2048, n = 2048, k = 2048;

  const std::size_t Asz = (std::size_t)m * k;
  const std::size_t Bsz = (std::size_t)k * n;
  const std::size_t Csz = (std::size_t)m * n;

  std::vector<double> A(Asz), B(Bsz), C(Csz, 0.0);
  std::mt19937_64 rng(42);
  std::uniform_real_distribution<double> ud(-0.5, 0.5);
  for (auto &x : A)
    x = ud(rng);
  for (auto &x : B)
    x = ud(rng);

  const double flop = 2.0 * (double)m * n * k;

  double t0 = omp_get_wtime();

  dgemm_naive(A.data(), B.data(), C.data(), m, n, k);

  double t1 = omp_get_wtime();

  double gflops = (flop / (t1 - t0)) * 1e-9;
  std::printf("Time for %d threads: %.6f s,   Rate: %.2f GFLOP/s\n",
              NUM_THREADS, (t1 - t0), gflops);

  int L = 2048, D = 1024;

  const std::size_t Qsz = (std::size_t)L * D;
  const std::size_t Ksz = (std::size_t)L * D;
  const std::size_t Vsz = (std::size_t)L * D;
  const std::size_t Osz = (std::size_t)L * D;

  std::vector<double> Q(Qsz), K(Ksz), V(Vsz), Ogem(Osz, 0.0), Od(Osz, 0.0);
  for (auto &x : Q)
    x = ud(rng);
  for (auto &x : K)
    x = ud(rng);
  for (auto &x : V)
    x = ud(rng);

  t0 = omp_get_wtime();

  attention_via_dgemm(Q.data(), K.data(), V.data(), Ogem.data(), L, D, 0, 0, 0,
                      0);

  t1 = omp_get_wtime();

  gflops = (flop / (t1 - t0)) * 1e-9;
  std::printf("Time for attention_via_dgemm: %.6f s,   Rate: %.2f GFLOP/s\n",
              (t1 - t0), gflops);

  t0 = omp_get_wtime();
  attention_direct(Q.data(), K.data(), V.data(), Od.data(), L, D);
  t1 = omp_get_wtime();

  gflops = (flop / (t1 - t0)) * 1e-9;
  std::printf("Time for attention_direct: %.6f s,   Rate: %.2f GFLOP/s\n",
              (t1 - t0), gflops);

  for (int i = 0; i < L; i++) {
    for (int j = 0; j < D; j++) {
      double g = Ogem[i * D + j];
      double d = Od[i * D + j];
      // Has a tolerance of 10%
      if ((g - d) / g > 0.1) {
        printf("Error in results: %lf %lf not matching at %d %d\n",
               Ogem[i * D + j], Od[i * D + j], i, j);
      }
    }
  }

  // try {
  // verify_dgemm_vs_mkl(A.data(), B.data(), C.data(), m, n, k);
  // } catch (const std::exception &e) {
  // std::printf("[verify] %s\n", e.what());
  // }

  // similarly call other functions
  return 0;
}
