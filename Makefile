# Compiler
CXX = g++
CXXFLAGS = -Ilibs -Wall -std=c++17

# Build rule
all: serial_canny mpi_canny omp_canny cuda_canny

serial_canny: serial_canny.cpp libs/png_read_write.cpp
	g++ -Ilibs -Wall -std=c++17 serial_canny.cpp libs/png_read_write.cpp -o serial_canny.exe

mpi_canny: mpi_canny.cpp libs/png_read_write.cpp
	mpic++ -Ilibs -Wall -std=c++17 mpi_canny.cpp libs/png_read_write.cpp -o mpi_canny.exe

omp_canny: omp_canny.cpp libs/png_read_write.cpp
	g++ -Ilibs -Wall -std=c++17 omp_canny.cpp libs/png_read_write.cpp -o omp_canny.exe -fopenmp

cuda_canny: cuda_canny.cu
	nvcc -Ilibs -std=c++17 libs/png_read_write.cpp cuda_canny.cu -o cuda_canny

run_canny: serial_canny
	./serial_canny.exe

run_canny_mpi: mpi_canny
	mpirun -np 4 ./mpi_canny.exe

run_canny_omp: omp_canny
	./omp_canny.exe

run_cuda_canny: cuda_canny
	./cuda_canny

clean:
	rm -f serial_canny.exe
	rm -f mpi_canny.exe
	rm -f omp_canny.exe
	rm -f cuda_canny.exe