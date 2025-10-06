#include "summa.hpp"
#include "localmatrix.hpp"
#include "verify.hpp"
#include <iostream>

extern void local_dgemm_cpu(LocalMatrix& C, const double* Arow, int Am, int An,
                            const double* Bcol, int Bm, int Bn);

void run_summa_cpu(int N,const Dist2D& d, bool do_verify){
  LocalMatrix A(N,N,d),B(N,N,d),C(N,N,d);
  A.initialize_A();
  B.initialize_B();
  C.zero();


  if(do_verify){
    // verify result
  }
}
