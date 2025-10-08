#include "common.hpp"
#include "summa.hpp"
#include <ctime>
#include <string>

int main(int argc, char **argv)
{
  MPI_Init(&argc, &argv);
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int N = 512;
  bool gpu = false;
  bool verify = false;
  for (int i = 1; i < argc; i++)
  {
    std::string arg = argv[i];
    if (arg == "--N" && i + 1 < argc)
      N = std::stoi(argv[++i]);
    else if (arg == "--gpu")
      gpu = true;
    else if (arg == "--verify")
      verify = true;
  }
  if (rank == 0)
    std::cout << "Opts: N=" << N << " P=" << size << " mode=" << (gpu ? "gpu" : "cpu") << " verify=" << verify << std::endl;

  Dist2D d;
  create_square_grid(size, rank, d); // Each process has its own grid info in d

  std::clock_t start = std::clock();

  if (gpu)
  {
#ifdef ENABLE_CUDA
    run_summa_gpu(N, d, verify);
#else
    if (rank == 0)
      std::cerr << "GPU not built (compile with nvcc and -DENABLE_CUDA)." << std::endl;
    MPI_Abort(MPI_COMM_WORLD, 1);
#endif
  }
  else
  {
    run_summa_cpu(N, d, verify);
  }

  std::clock_t end = std::clock();
  double elapsed_secs = double(end - start) / CLOCKS_PER_SEC;

  std::cout << "Time: " << elapsed_secs << "s\n";

  MPI_Comm_free(&d.row_comm);
  MPI_Comm_free(&d.col_comm);
  MPI_Finalize();
}
