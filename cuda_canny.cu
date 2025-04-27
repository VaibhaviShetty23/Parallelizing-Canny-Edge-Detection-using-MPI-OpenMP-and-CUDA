#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <sstream>
#include "png_read_write.h"    // Your PNG reader/writer
#include <chrono>
#include <cuda_runtime.h>

using namespace std;
using namespace std::chrono;

const bool DEBUG = false;

// Error checking macro
#define CHECK_CUDA(call)                                                     \
    {                                                                        \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                            \
            cerr << "CUDA error: " << cudaGetErrorString(err)                \
                 << " at line " << __LINE__ << endl;                         \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    }

//-------------------------------------------------------------
// Host helper function: logs a matrix (for debugging)
void logMatrix(const vector<vector<int>>& matrix, ofstream& logfile) {
    for (const auto& row : matrix) {
        for (int val : row) {
            logfile << val << " ";
        }
        logfile << "\n";
    }
}

//-------------------------------------------------------------
// Constant memory for 5x5 Gaussian kernel:
__constant__ float d_gaussKernel[5][5];
// Constant memory for the Sobel kernels:
__constant__ int d_gx[3][3];
__constant__ int d_gy[3][3];

//-------------------------------------------------------------
// CUDA kernel for applying Gaussian Blur.
__global__ void gaussianBlurKernel(const int* input, int* output, int width, int height) {
    // Use 2 as the kernel radius
    int kernelRadius = 2;
    int x = blockIdx.x * blockDim.x + threadIdx.x; // column
    int y = blockIdx.y * blockDim.y + threadIdx.y; // row

    if (x < width && y < height) {
        // Only process pixels where the kernel fits, else pass through
        if (x >= kernelRadius && x < width - kernelRadius &&
            y >= kernelRadius && y < height - kernelRadius) {
            float sum = 0.0f;
            for (int ky = -kernelRadius; ky <= kernelRadius; ++ky) {
                for (int kx = -kernelRadius; kx <= kernelRadius; ++kx) {
                    int ix = x + kx;
                    int iy = y + ky;
                    sum += input[iy * width + ix] * d_gaussKernel[ky + kernelRadius][kx + kernelRadius];
                }
            }
            output[y * width + x] = int(roundf(sum));
        } else {
            // Border pixels; keep original value
            output[y * width + x] = input[y * width + x];
        }
    }
}

//-------------------------------------------------------------
// CUDA kernel for computing gradients using Sobel kernels.
__global__ void computeGradientsKernel(const int* input, int* gradX, int* gradY, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        if (x >= 1 && x < width - 1 && y >= 1 && y < height - 1) {
            int sumX = 0, sumY = 0;
            for (int ky = -1; ky <= 1; ++ky) {
                for (int kx = -1; kx <= 1; ++kx) {
                    int ix = x + kx;
                    int iy = y + ky;
                    int pixel = input[iy * width + ix];
                    sumX += pixel * d_gx[ky + 1][kx + 1];
                    sumY += pixel * d_gy[ky + 1][kx + 1];
                }
            }
            gradX[y * width + x] = sumX;
            gradY[y * width + x] = sumY;
        } else {
            gradX[y * width + x] = 0;
            gradY[y * width + x] = 0;
        }
    }
}

//-------------------------------------------------------------
// CUDA kernel for computing magnitude and direction.
__global__ void computeMagnitudeDirectionKernel(const int* gradX, const int* gradY, int* magnitude, int* direction, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        if (x >= 1 && x < width - 1 && y >= 1 && y < height - 1) {
            int gx = gradX[y * width + x];
            int gy = gradY[y * width + x];
            int mag = int(round(sqrtf((float)(gx * gx + gy * gy))));
            int dir = int(round(atan2f((float)gy, (float)gx) * 180.0f / 3.14159265f));
            magnitude[y * width + x] = mag;
            direction[y * width + x] = dir;
        } else {
            magnitude[y * width + x] = 0;
            direction[y * width + x] = 0;
        }
    }
}

//-------------------------------------------------------------
// CUDA kernel for non-maximum suppression.
__global__ void nonMaxSuppKernel(const int* magnitude, const int* direction, int* suppressed, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        if (x >= 1 && x < width - 1 && y >= 1 && y < height - 1) {
            int idx = y * width + x;
            int angle = direction[idx] % 180;
            if (angle < 0)
                angle += 180;
            int current = magnitude[idx];
            int neighbor1 = 0, neighbor2 = 0;
            
            if ((angle >= 0 && angle < 22.5) || (angle >= 157.5 && angle < 180)) {
                neighbor1 = magnitude[y * width + (x - 1)];
                neighbor2 = magnitude[y * width + (x + 1)];
            }
            else if (angle >= 22.5 && angle < 67.5) {
                neighbor1 = magnitude[(y - 1) * width + (x + 1)];
                neighbor2 = magnitude[(y + 1) * width + (x - 1)];
            }
            else if (angle >= 67.5 && angle < 112.5) {
                neighbor1 = magnitude[(y - 1) * width + x];
                neighbor2 = magnitude[(y + 1) * width + x];
            }
            else if (angle >= 112.5 && angle < 157.5) {
                neighbor1 = magnitude[(y - 1) * width + (x - 1)];
                neighbor2 = magnitude[(y + 1) * width + (x + 1)];
            }
            suppressed[idx] = (current < neighbor1 || current < neighbor2) ? 0 : current;
        } else {
            suppressed[y * width + x] = magnitude[y * width + x];
        }
    }
}

//-------------------------------------------------------------
// CUDA kernel for double thresholding.
__global__ void doubleThresholdKernel(const int* suppressed, int* edge, int width, int height, int lowThresh, int highThresh) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        int idx = y * width + x;
        int value = suppressed[idx];
        if (value >= highThresh)
            edge[idx] = 255;
        else if (value >= lowThresh)
            edge[idx] = 128;
        else
            edge[idx] = 0;
    }
}

//-------------------------------------------------------------
// CUDA kernel for edge tracking by hysteresis.
__global__ void edgeTrackingKernel(int* edge, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= 1 && x < width - 1 && y >= 1 && y < height - 1) {
        int idx = y * width + x;
        if (edge[idx] == 128) {
            if (edge[(y - 1) * width + (x - 1)] == 255 ||
                edge[(y - 1) * width + x] == 255 ||
                edge[(y - 1) * width + (x + 1)] == 255 ||
                edge[y * width + (x - 1)] == 255 ||
                edge[y * width + (x + 1)] == 255 ||
                edge[(y + 1) * width + (x - 1)] == 255 ||
                edge[(y + 1) * width + x] == 255 ||
                edge[(y + 1) * width + (x + 1)] == 255)
                edge[idx] = 255;
            else
                edge[idx] = 0;
        }
    }
}

//-------------------------------------------------------------
// Helper: set up and launch kernels with a 16x16 thread block.
dim3 getBlockDim() {
    return dim3(16, 16);
}
dim3 getGridDim(int width, int height, dim3 blockDim) {
    return dim3((width + blockDim.x - 1) / blockDim.x,
                (height + blockDim.y - 1) / blockDim.y);
}

//-------------------------------------------------------------
// Main host function.
int main() {
    // Input/Output filenames (adjust as needed)
    string input_filename = "input/emoji.png";
    string output_filename = "output/emoji_cuda.png";
    
    // Read image from file using your PNG functions.
    vector<vector<int>> image = read_png(input_filename);
    if (image.empty()) {
        cerr << "Error reading image!" << endl;
        return -1;
    }
    
    int height = image.size();
    int width = image[0].size();
    
    
    // Start timer.
    auto start = high_resolution_clock::now();
    
    // Flatten the 2D image into a 1D vector.
    vector<int> h_input(width * height);
    for (int i = 0; i < height; ++i)
        for (int j = 0; j < width; ++j)
            h_input[i * width + j] = image[i][j];
    
    // Allocate space for the intermediate results.
    vector<int> h_blurred(width * height);
    vector<int> h_gradX(width * height);
    vector<int> h_gradY(width * height);
    vector<int> h_magnitude(width * height);
    vector<int> h_direction(width * height);
    vector<int> h_suppressed(width * height);
    vector<int> h_edge(width * height);
    
    // Allocate device memory.
    int *d_input, *d_blurred, *d_gradX, *d_gradY, *d_magnitude, *d_direction, *d_suppressed, *d_edge;
    size_t numBytes = width * height * sizeof(int);
    CHECK_CUDA(cudaMalloc(&d_input, numBytes));
    CHECK_CUDA(cudaMalloc(&d_blurred, numBytes));
    CHECK_CUDA(cudaMalloc(&d_gradX, numBytes));
    CHECK_CUDA(cudaMalloc(&d_gradY, numBytes));
    CHECK_CUDA(cudaMalloc(&d_magnitude, numBytes));
    CHECK_CUDA(cudaMalloc(&d_direction, numBytes));
    CHECK_CUDA(cudaMalloc(&d_suppressed, numBytes));
    CHECK_CUDA(cudaMalloc(&d_edge, numBytes));
    
    // Copy the original image to device.
    CHECK_CUDA(cudaMemcpy(d_input, h_input.data(), numBytes, cudaMemcpyHostToDevice));
    
    // Set up host copies for the constant memory kernels.
    const float h_gaussKernel[5][5] = {
        {1.0f/273, 4.0f/273, 7.0f/273, 4.0f/273, 1.0f/273},
        {4.0f/273, 16.0f/273, 26.0f/273, 16.0f/273, 4.0f/273},
        {7.0f/273, 26.0f/273, 41.0f/273, 26.0f/273, 7.0f/273},
        {4.0f/273, 16.0f/273, 26.0f/273, 16.0f/273, 4.0f/273},
        {1.0f/273, 4.0f/273, 7.0f/273, 4.0f/273, 1.0f/273}
    };
    const int h_gx[3][3] = {
        {-1, 0, 1},
        {-2, 0, 2},
        {-1, 0, 1}
    };
    const int h_gy[3][3] = {
        {-1, -2, -1},
        { 0,  0,  0},
        { 1,  2,  1}
    };
    
    // Copy kernels to device constant memory.
    CHECK_CUDA(cudaMemcpyToSymbol(d_gaussKernel, h_gaussKernel, sizeof(h_gaussKernel)));
    CHECK_CUDA(cudaMemcpyToSymbol(d_gx, h_gx, sizeof(h_gx)));
    CHECK_CUDA(cudaMemcpyToSymbol(d_gy, h_gy, sizeof(h_gy)));
    
    dim3 blockDim = getBlockDim();
    dim3 gridDim = getGridDim(width, height, blockDim);
    
    // --- Stage 1: Gaussian Blur ---
    gaussianBlurKernel<<<gridDim, blockDim>>>(d_input, d_blurred, width, height);
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // --- Stage 2: Compute Gradients ---
    computeGradientsKernel<<<gridDim, blockDim>>>(d_blurred, d_gradX, d_gradY, width, height);
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // --- Stage 3: Compute Magnitude and Direction ---
    computeMagnitudeDirectionKernel<<<gridDim, blockDim>>>(d_gradX, d_gradY, d_magnitude, d_direction, width, height);
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // --- Stage 4: Non-Maximum Suppression ---
    nonMaxSuppKernel<<<gridDim, blockDim>>>(d_magnitude, d_direction, d_suppressed, width, height);
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // --- Stage 5: Double Thresholding ---
    // Example thresholds: low = 30, high = 100.
    doubleThresholdKernel<<<gridDim, blockDim>>>(d_suppressed, d_edge, width, height, 30, 100);
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // --- Stage 6: Edge Tracking by Hysteresis ---
    edgeTrackingKernel<<<gridDim, blockDim>>>(d_edge, width, height);
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // End timer.
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(end - start);
    cout << "Execution time: " << duration.count() << " ms" << endl;
    
    // Copy final result back to host.
    CHECK_CUDA(cudaMemcpy(h_edge.data(), d_edge, numBytes, cudaMemcpyDeviceToHost));
    
    // (Optional) Convert the flattened h_edge back into a 2D vector for your PNG writer.
    vector<vector<int>> edge2D(height, vector<int>(width));
    for (int i = 0; i < height; ++i)
        for (int j = 0; j < width; ++j)
            edge2D[i][j] = h_edge[i * width + j];
    
    // Write result to file.
    write_png(output_filename, edge2D, width, height);
    
    // (Optional) Logging in debug mode.
    if (DEBUG) {
        ofstream logfile("logs/cuda.txt", ios::out | ios::trunc);
        logfile << "Final Edge Map:\n";
        logMatrix(edge2D, logfile);
        logfile.close();
    }
    
    // Free device memory.
    cudaFree(d_input);
    cudaFree(d_blurred);
    cudaFree(d_gradX);
    cudaFree(d_gradY);
    cudaFree(d_magnitude);
    cudaFree(d_direction);
    cudaFree(d_suppressed);
    cudaFree(d_edge);
    
    return 0;
}
