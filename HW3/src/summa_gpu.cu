#include "summa.hpp"
#include "localmatrix.hpp"
#include "verify.hpp"
#ifdef ENABLE_CUDA
#include <cuda_runtime.h>
#include <vector>
#include <iostream>

extern void local_dgemm_gpu(double* dC,int Cm,int Cn,
                            const double* dA,int Am,int An,
                            const double* dB,int Bm,int Bn,
                            cudaStream_t stream);

void run_summa_gpu(int N,const Dist2D& d, bool do_verify){
  LocalMatrix A(N,N,d),B(N,N,d),C(N,N,d);
  A.initialize_A();
  B.initialize_B();
  C.zero();

}
#endif
