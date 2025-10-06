#include "verify.hpp"
#include <mpi.h>
#include <iostream>
#include <cmath>

void gather_matrix(const LocalMatrix& C, int N, const Dist2D& d,
                   std::vector<double>& fullC, int root){
    int rank; MPI_Comm_rank(MPI_COMM_WORLD,&rank);

}

void verify_result(int N, const std::vector<double>& fullC){
}
