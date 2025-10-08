#pragma once
#include <vector>
#include "common.hpp"

class LocalMatrix
{
public:
    int g_rows, g_cols; // global rows and columns
    int l_rows, l_cols; // local rows and columns
    int row_off, col_off;
    Dist2D grid;
    std::vector<double> data; // row-major: data[i*l_cols + j]

    LocalMatrix(int g_m, int g_n, const Dist2D &d)
        : g_rows(g_m), g_cols(g_n), grid(d)
    {
        int row_rem = g_m % d.P;
        l_rows = g_m / d.P + (d.myr < row_rem ? 1 : 0);

        int col_rem = g_n % d.P;
        l_cols = g_n / d.P + (d.myc < col_rem ? 1 : 0);

        row_off = d.myr * l_rows;
        col_off = d.myc * l_cols;

        data.resize(l_rows * l_cols);
    }

    double &operator()(int i, int j) { return data[i * l_cols + j]; }
    const double &operator()(int i, int j) const { return data[i * l_cols + j]; }

    void zero()
    {
        for (int i = 0; i < l_rows * l_cols; ++i)
            data[i] = 0.0;
    }

    // Deterministic initializers (simple integers; no comm needed)
    void initialize_A()
    {
        // A is row-major
        for (int i = 0; i < l_rows; ++i)
            for (int j = 0; j < l_cols; ++j)
                (*this)(i, j) = (row_off + i) + (col_off + j);
    }
    void initialize_B()
    {
        // B is col-major
        for (int i = 0; i < l_rows; ++i)
            for (int j = 0; j < l_cols; ++j)
                data[i + j * l_rows] = (row_off + i) - (col_off + j);
    }
};
