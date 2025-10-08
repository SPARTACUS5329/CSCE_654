#include "localmatrix.hpp"

// Write your local DGEMM usign OpenMP
void local_dgemm_cpu(double *Crow, const double *Arow, int Am, int An,
                     const double *Bcol, int Bm, int Bn)
{
    // An == Bm
    // C (Am x Bn)
    if (An != Bm)
    {
        perror("Am != Bn");
        exit(1);
    }

#pragma omp parallel for schedule(dynamic, 8) num_threads(NUM_THREADS)
    for (int i = 0; i < Am; i++)
    {
        for (int j = 0; j < An; j++)
        {
#pragma omp simd
            for (int k = 0; k < An; k++)
            {
                // C(Am * Bn) An == Bm
                Crow[i * Bn + j] += Arow[i * An + k] * Bcol[k + j * Bm];
            }
        }
    }
}
