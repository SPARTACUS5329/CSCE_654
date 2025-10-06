#include "localmatrix.hpp"
#ifdef ENABLE_CUDA
#include <cuda_runtime.h>



// Write your local DGEMM usign Cuda
void local_dgemm_gpu(double* dC,int Cm,int Cn,
                     const double* dA,int Am,int An,
                     const double* dB,int Bm,int Bn,
                     cudaStream_t stream=0){

}
#endif
