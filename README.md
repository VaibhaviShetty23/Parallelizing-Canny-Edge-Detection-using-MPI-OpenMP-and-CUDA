![C++](https://img.shields.io/badge/language-C++-blue.svg)
![CUDA](https://img.shields.io/badge/GPU-CUDA-green.svg)
![OpenMP](https://img.shields.io/badge/Parallel-OpenMP-yellow.svg)
![MPI](https://img.shields.io/badge/Distributed-MPI-orange.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)

# Parallelizing-Canny-Edge-Detection

This project implements the Canny Edge Detection algorithm using three parallel programming models: MPI, OpenMP, and CUDA. The serial version of the algorithm was first developed step-by-step, including Gaussian smoothing, gradient calculation, non-maximum suppression, double thresholding, and edge tracking by hysteresis. Each stage was carefully analyzed to identify pixel-level data parallelism that could be exploited to enhance computational performance.

The parallel implementations use different strategies tailored to the architecture: MPI distributes pixels across multiple processes for execution in a distributed memory environment, OpenMP applies multithreaded parallelism on a shared-memory multicore CPU, and CUDA accelerates processing by mapping pixels to thousands of GPU threads. Results demonstrate significant performance improvements across all methods, with CUDA delivering the highest speedup, particularly for large-scale images.

## Technologies Used

- **C++** — Implemented the serial, MPI, and OpenMP versions of Canny Edge Detection.
- **CUDA C/C++** — Developed GPU-accelerated parallelization using NVIDIA GPUs.
- **MPI (Message Passing Interface)** — Used for distributed-memory parallel computing across processes.
- **OpenMP (Open Multi-Processing)** — Enabled shared-memory parallelism with multithreading on multicore CPUs.
- **NVIDIA CUDA Toolkit** — Compiled and executed CUDA programs for GPU acceleration.
- **GCC / G++** — Compilers used for building C++ and OpenMP code.
- **OpenMPI** — Library used to compile and run MPI-based parallel programs.
- **Python (optional)** — Utilized for plotting performance graphs and visualizing execution time comparisons.


## Steps to run

1. Checkout the code on local.
2. On the root run `make` to create executables for serial, OpenMP, MPI and CUDA implementation.
3. Run `make run_canny` to run the serial implementation, `make run_canny_mpi` to run the MPI implementation, `make run_canny_omp` for the OpenMP implementation and `make run_cuda_canny` for CUDA. The code is set to run each implementation on `input/emoji.png`.
4. To verify the results check the `output` directory.

## Optional

Change the input image by changing the `filename` variable in the main function of each implementation.

# Contributors

1. Vaibhavi Shetty - vshetty2@ncsu.edu
2. Swaraj Kaondal - skaonda@ncsu.edu
