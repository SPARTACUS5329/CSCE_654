#pragma once
#include "localmatrix.hpp"
#include <vector>

void gather_matrix(const LocalMatrix &A, int N, std::vector<double> &fullA, int root);
void verify_result(int N, const std::vector<double> A, const std::vector<double> B, const std::vector<double> &fullC);
void chunks_to_row_major(const std::vector<double> &A, std::vector<double> &convertedA, int N, int P, int l_rows, int l_cols);
void chunks_to_col_major(const std::vector<double> &A, std::vector<double> &convertedA, int N, int P, int l_rows, int l_cols);
