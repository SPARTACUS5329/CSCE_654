#include "summa.hpp"
#include "localmatrix.hpp"
#include "verify.hpp"
#include <iostream>

void run_summa_cpu(int N, const Dist2D &d, bool do_verify)
{
  LocalMatrix A(N, N, d), B(N, N, d), C(N, N, d);
  A.initialize_A();
  B.initialize_B();
  C.zero();

  std::vector<double> Aik(A.l_rows * A.l_cols);
  std::vector<double> Bkj(B.l_rows * B.l_cols);

  for (int k = 0; k < d.P; k++)
  {
    if (d.myc == k)
      for (int i = 0; i < A.l_rows * A.l_cols; i++)
        Aik[i] = A.data[i];

    if (d.myr == k)
      for (int i = 0; i < B.l_rows * B.l_cols; i++)
        Bkj[i] = B.data[i];

    MPI_Bcast(Aik.data(), A.l_rows * A.l_cols, MPI_DOUBLE, k, d.row_comm);
    MPI_Bcast(Bkj.data(), B.l_rows * B.l_cols, MPI_DOUBLE, k, d.col_comm);

    local_dgemm_cpu(C.data.data(), Aik.data(), A.l_rows, A.l_cols, Bkj.data(), B.l_rows, B.l_cols);
  }

  if (do_verify)
  {
    std::vector<double> fullA(N * N);
    std::vector<double> fullB(N * N);
    std::vector<double> fullC(N * N);

    gather_matrix(A, N, fullA, 0);
    gather_matrix(B, N, fullB, 0);
    gather_matrix(C, N, fullC, 0);

    std::vector<double> convA(N * N);
    std::vector<double> convB(N * N);
    std::vector<double> convC(N * N);

    if (d.rank == 0)
    {
      chunks_to_row_major(fullA, convA, N, d.P, A.l_rows, A.l_cols);
      chunks_to_col_major(fullB, convB, N, d.P, B.l_rows, B.l_cols);
      chunks_to_row_major(fullC, convC, N, d.P, C.l_rows, C.l_cols);
      verify_result(N, convA, convB, convC);
    }
  }
}
