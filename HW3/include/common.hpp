#pragma once
#include <mpi.h>
#include <cmath>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <algorithm>

struct Dist2D
{
  int P;
  int myr, myc;
  int rank;
  MPI_Comm row_comm, col_comm;
};

inline void create_square_grid(int world_size, int rank, Dist2D &d)
{
  int P = std::sqrt(world_size);
  if (P * P != world_size)
  {
    if (rank == 0)
      std::cerr << "Error: world_size must be a perfect square (got " << world_size << ")" << std::endl;
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  d.P = P;
  d.myr = rank / P;
  d.myc = rank % P;
  d.rank = rank;
  MPI_Comm_split(MPI_COMM_WORLD, d.myr, d.myc, &d.row_comm);
  MPI_Comm_split(MPI_COMM_WORLD, d.myc, d.myr, &d.col_comm);
}

// Split N items across P parts: sizes[r], offs[r]
// You do not have to use this function.
// This is just a suggestion
inline void split_sizes(int N, int P, std::vector<int> &sizes, std::vector<int> &offs)
{
  sizes.assign(P, N / P);
  int rem = N % P;
  for (int r = 0; r < P; r++)
    if (r < rem)
      sizes[r]++;
  offs.assign(P, 0);
  for (int r = 1; r < P; r++)
    offs[r] = offs[r - 1] + sizes[r - 1];
}

void local_dgemm_cpu(double *Crow, const double *Arow, int Am, int An,
                     const double *Bcol, int Bm, int Bn);

void local_dgemm_gpu(double *dC,
                     const double *dA, int Am, int An,
                     const double *dB, int Bm, int Bn,
                     cudaStream_t stream);