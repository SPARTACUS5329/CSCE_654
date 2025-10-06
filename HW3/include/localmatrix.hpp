#pragma once
#include <vector>
#include "common.hpp"

class LocalMatrix {
public:
    int g_rows, g_cols; // global rows and columns
    int l_rows, l_cols; // local rows and columns
    Dist2D grid;
    std::vector<double> data;  // row-major: data[i*l_cols + j]

    LocalMatrix(int g_m,int g_n,const Dist2D& d)
        : g_rows(g_m), g_cols(g_n), grid(d)
    {
       // Populate necessary data
    }

    double& operator()(int i,int j){ return data[i*l_cols+j]; }
    const double& operator()(int i,int j) const { return data[i*l_cols+j]; }

    void zero(){ }

    // Deterministic initializers (simple integers; no comm needed)
    void initialize_A(){
    }
    void initialize_B(){
    }
};
