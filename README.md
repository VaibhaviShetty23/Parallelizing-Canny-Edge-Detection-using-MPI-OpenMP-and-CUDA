![C++](https://img.shields.io/badge/language-C++-blue.svg)
![CUDA](https://img.shields.io/badge/GPU-CUDA-green.svg)
![OpenMP](https://img.shields.io/badge/Parallel-OpenMP-yellow.svg)
![MPI](https://img.shields.io/badge/Distributed-MPI-orange.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)

# Parallelizing-Canny-Edge-Detection

This project implements Canny Edge Detection using parallel programming models: MPI, OpenMP, and CUDA.
The serial algorithm was first built step-by-step and then parallelized at the pixel level for improved performance.
MPI distributes pixels across processes, OpenMP uses multi-threaded CPU execution, and CUDA leverages GPU acceleration for massive speedup.
Results demonstrate significant performance gains, especially with CUDA.

# Steps to run

1. Checkout the code on local.
2. On the root run `make` to create executables for serial, OpenMP, MPI and CUDA implementation.
3. Run `make run_canny` to run the serial implementation, `make run_canny_mpi` to run the MPI implementation, `make run_canny_omp` for the OpenMP implementation and `make run_cuda_canny` for CUDA. The code is set to run each implementation on `input/emoji.png`.
4. To verify the results check the `output` directory.

## Optional

Change the input image by changing the `filename` variable in the main function of each implementation.

# Contributors

1. Vaibhavi Shetty - vshetty2@ncsu.edu
2. Swaraj Kaondal - skaonda@ncsu.edu
