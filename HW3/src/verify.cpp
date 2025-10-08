#include "verify.hpp"
#include "localmatrix.hpp"
#include "common.hpp"
#include <mpi.h>
#include <iostream>
#include <cmath>

void gather_matrix(const LocalMatrix &A, int N, std::vector<double> &fullA, int root)
{
    MPI_Gather(A.data.data(), A.l_rows * A.l_cols, MPI_DOUBLE, fullA.data(), A.l_rows * A.l_cols, MPI_DOUBLE, root, MPI_COMM_WORLD);
}

void chunks_to_row_major(const std::vector<double> &A, std::vector<double> &convertedA, int N, int P, int l_rows, int l_cols)
{
    for (int k = 0; k < P * P; k++)
    {
        int p_row = k / P;
        int p_col = k % P;
        for (int i = 0; i < l_rows; i++)
        {
            for (int j = 0; j < l_cols; j++)
            {
                int global_row = p_row * l_rows + i;
                int global_col = p_col * l_cols + j;

                convertedA[global_row * N + global_col] =
                    A[k * l_rows * l_cols + i * l_cols + j];
            }
        }
    }
}

void chunks_to_col_major(const std::vector<double> &A, std::vector<double> &convertedA, int N, int P, int l_rows, int l_cols)
{
    for (int k = 0; k < P * P; k++)
    {
        int p_row = k / P;
        int p_col = k % P;
        for (int i = 0; i < l_cols; i++)
        {
            for (int j = 0; j < l_rows; j++)
            {
                int global_col = p_col * l_cols + i;
                int global_row = p_row * l_rows + j;

                convertedA[global_col * N + global_row] =
                    A[k * l_rows * l_cols + i * l_cols + j];
            }
        }
    }
}

void verify_result(int N, const std::vector<double> A, const std::vector<double> B, const std::vector<double> &fullC)
{
    std::vector<double> C(N * N);
    local_dgemm_cpu(C.data(), A.data(), N, N, B.data(), N, N);
    for (int i = 0; i < N * N; i++)
    {
        if (C[i] != fullC[i])
        {
            double err = (C[i] - fullC[i]) / C[i];
            printf("Matrices don't match err: %lf\n", err);
            return;
        }
    }
    printf("Matrices match!\n");
}
