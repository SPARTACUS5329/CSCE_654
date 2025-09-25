/*
We will compute the covariance between two large vectors of numbers in parallel.
The vectors are randomly initialized by the root process.
Because the vectors are large, the work is divided among multiple processes.
Each process will receive a chunk of the vectors, compute partial results, and
then combine them with the results from the other processes. The root process is
responsible to print the final output.
*/

#include <cstdlib>
#include <ctime>
#include <iostream>
#include <mpi.h>
#include <omp.h>
#include <vector>

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);

  int myrank, p;
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
  MPI_Comm_size(MPI_COMM_WORLD, &p);

  // --- Step 1: Parse command line argument ---
  // Only the root processes perform this
  int n = 0;
  if (myrank == 0) {
    if (argc < 2) {
      std::cerr << "Usage: mpirun -np <p> ./a.out <vector_length>" << std::endl;
      n = 0;
    } else {
      n = std::atoi(argv[1]);
    }

    if (n <= 1) {
      std::cerr << "Vector length must be > 1" << std::endl;
      MPI_Finalize();
      return 1;
    }
  }

  // --- Step 2: Root initializes full vectors ---
  std::vector<double> x_global, y_global;
  if (myrank == 0) {
    x_global.resize(n);
    y_global.resize(n);
    srand(time(NULL));
    for (int i = 0; i < n; i++) {
      x_global[i] = (double)rand() / RAND_MAX; // [0,1]
      y_global[i] = (double)rand() / RAND_MAX;
    }
  }

  // --- Step 3: Root computes counts and displacements based on a data
  // distribution policy If n%p=0, each process should get n/p elements If
  // n%p!=0, some processes will get 1 more enetry Any distribution is fine, but
  // this is load-balanced Each process can count it if they know the
  // distribution policy
  std::vector<int> counts, displs;
  if (myrank == 0) {
    counts.resize(p);
    displs.resize(p);

    int base = n / p;
    int rem = n % p;
    for (int i = 0; i < p; i++) {
      if (i < rem)
        counts[i] = base + 1;
      else
        counts[i] = base;
      displs[i] = (i == 0 ? 0 : displs[i - 1] + counts[i - 1]);
    }
  }

  MPI_Barrier(MPI_COMM_WORLD); // sync before timing
  double t_start = MPI_Wtime();

  // --- Step 4: Broadcast len to all processes

  void *sendbuf = counts.data();
  void *recvbuf = counts.data();

  if (myrank == 0) {
    MPI_Scatter(sendbuf, n, MPI_INT, MPI_IN_PLACE, 0, MPI_INT, myrank,
                MPI_COMM_WORLD);
  } else {
    MPI_Scatter(MPI_IN_PLACE, 0, MPI_INT, recvbuf, n, MPI_INT, myrank,
                MPI_COMM_WORLD);
  }

  // --- Step 5: allocate local buffers

  int n_local;
  if (myrank < (n % p))
    n_local = n / p + 1;
  else
    n_local = n / p;
  std::vector<double> x_local(n_local), y_local(n_local);

  // --- Step 6: Root create data and scatter to other processes

  /**** Write a scatter Calls here */
  if (myrank == 0) {
    MPI_Scatterv(&x_global, counts.data(), displs.data(), MPI_INT, recvbuf, 0,
                 MPI_INT, myrank, MPI_COMM_WORLD);

    MPI_Scatterv(&y_global, counts.data(), displs.data(), MPI_INT, recvbuf, 0,
                 MPI_INT, myrank, MPI_COMM_WORLD);
  } else {
    MPI_Scatterv(MPI_IN_PLACE, nullptr, nullptr, MPI_INT, x_local.data(),
                 n_local, MPI_INT, myrank, MPI_COMM_WORLD);

    MPI_Scatterv(MPI_IN_PLACE, nullptr, nullptr, MPI_INT, y_local.data(),
                 n_local, MPI_INT, myrank, MPI_COMM_WORLD);
  }

  // --- Step 7: Local sums ---
  double local_sum_x = 0.0, local_sum_y = 0.0;
#pragma omp parallel for schedule(static)                                      \
    reduction(+ : local_sum_x, local_sum_y)
  for (int i = 0; i < n_local; i++) {
    local_sum_x += x_local[i];
    local_sum_y += y_local[i];
  }

  // --- Step 8: Global means via allreduce ---
  double global_sum_x, global_sum_y;

  /**** Write a reductionall Call here */
  if (myrank == 0) {
    MPI_Allreduce(MPI_IN_PLACE, &global_sum_x, 1, MPI_DOUBLE, MPI_SUM,
                  MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &global_sum_y, 1, MPI_DOUBLE, MPI_SUM,
                  MPI_COMM_WORLD);
  } else {
    MPI_Allreduce(&local_sum_x, nullptr, 1, MPI_DOUBLE, MPI_SUM,
                  MPI_COMM_WORLD);
    MPI_Allreduce(&local_sum_y, nullptr, 1, MPI_DOUBLE, MPI_SUM,
                  MPI_COMM_WORLD);
  }

  double mean_x = global_sum_x / n;
  double mean_y = global_sum_y / n;

  // --- Step 9: Local covariance contribution ---
  double local_cov = 0.0;
#pragma omp parallel for reduction(+ : local_cov)
  for (int i = 0; i < n_local; i++) {
    local_cov += (x_local[i] - mean_x) * (y_local[i] - mean_y);
  }

  // --- Step 10: Reduce to root for printing ---
  double global_cov;

  /**** Write a reduction Call here */
  if (myrank == 0) {
    MPI_Allreduce(MPI_IN_PLACE, &global_cov, 1, MPI_DOUBLE, MPI_SUM,
                  MPI_COMM_WORLD);
  } else {
    MPI_Allreduce(&local_cov, nullptr, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  }

  MPI_Barrier(MPI_COMM_WORLD);
  double t_end = MPI_Wtime();

  if (myrank == 0) {
    global_cov /= (n - 1);
    std::cout << "Covariance = " << global_cov << std::endl;
    std::cout << "Elapsed time = " << (t_end - t_start) << " seconds"
              << std::endl;
  }

  MPI_Finalize();
  return 0;
}
