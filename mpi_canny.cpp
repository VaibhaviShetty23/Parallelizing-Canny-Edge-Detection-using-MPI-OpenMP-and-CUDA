#include <iostream>
#include <vector>
#define _USE_MATH_DEFINES
#include <cmath>
#include <fstream>
#include <sstream>
#include "png_read_write.h"
#include <chrono>
#include <mpi.h>
using namespace std;
using namespace std::chrono;
const bool DEBUG = false;

void logMatrix(const vector<vector<int>>& matrix, ofstream& logfile) {
    for (const auto& row : matrix) {
        for (int val : row) {
            logfile << val << " ";
        }
        logfile << "\n";
    }
}

const float kernel[5][5] = {
    {1.0f/273.0f, 4.0f/273.0f, 7.0f/273.0f, 4.0f/273.0f, 1.0f/273.0f},
    {4.0f/273.0f, 16.0f/273.0f, 26.0f/273.0f, 16.0f/273.0f, 4.0f/273.0f},
    {7.0f/273.0f, 26.0f/273.0f, 41.0f/273.0f, 26.0f/273.0f, 7.0f/273.0f},
    {4.0f/273.0f, 16.0f/273.0f, 26.0f/273.0f, 16.0f/273.0f, 4.0f/273.0f},
    {1.0f/273.0f, 4.0f/273.0f, 7.0f/273.0f, 4.0f/273.0f, 1.0f/273.0f}
};

// Function to apply Gaussian Blur
void gaussianBlur(const vector<vector<int>>& input, vector<vector<int>>& output, int width, int height, int world_size, int world_rank, ofstream& logfile) {
    int elements_per_process = ceil((height - 4)*(width - 4) / world_size);
    vector<int> local_array(elements_per_process, 0);
    int start = 2*width + 2;
    int offset = (world_rank * elements_per_process)/(width-4);
    int start_index = start + world_rank*elements_per_process + offset*4;

    int i = start_index;
    for(int j = 0; j < elements_per_process; ++j) {
        if((i - (width - 2)) % width == 0){
            i += 4;
        }
        float sum = 0;
        int x = i % width;
        int y = i / width;
        for (int ki = -2; ki <= 2; ++ki) {
            for (int kj = -2; kj <= 2; ++kj) {
                sum += input[y + ki][x + kj] * kernel[ki + 2][kj + 2];
            }
        }
        local_array[j] = static_cast<int>(round(sum));
        i += 1;
    }

    vector<int> gathered_array(world_size * elements_per_process);
    MPI_Allgather(local_array.data(), elements_per_process, MPI_INT, gathered_array.data(), elements_per_process, MPI_INT, MPI_COMM_WORLD);

    int idx = start;
    for (int j = 0; j < ((height - 4)*(width - 4)); ++j) {
        if((idx - (width - 2)) % width == 0){
            idx += 4;
        }
        int x = idx % width;
        int y = idx / width;
        output[y][x] = gathered_array[j];
        idx += 1;
    }
    if (world_rank == 0 && DEBUG) {
        logfile << "Gaussian Blur applied:\n";
        logMatrix(output, logfile);
        logfile << "\n";
    }
}

// Function to compute gradients
void computeGradients(const vector<vector<int>>& input, vector<vector<int>>& gradX, vector<vector<int>>& gradY, int width, int height, int world_size, int world_rank, ofstream& logfile) {
    int gx[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    int gy[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};

    int elements_per_process = ceil((height - 2)*(width - 2) / world_size);
    vector<int> local_x(elements_per_process, 0);
    vector<int> local_y(elements_per_process, 0);
    int start = width + 1;
    int offset = (world_rank * elements_per_process)/(width-2);
    int start_index = start + world_rank*elements_per_process + offset*2;
    
    int i = start_index;
    for(int j = 0; j < elements_per_process; ++j) {
        if((i - (width - 1)) % width == 0){
            i += 2;
        }
        int sumX = 0, sumY = 0;
        int x = i % width;
        int y = i / width;
        for (int ki = -1; ki <= 1; ++ki) {
            for (int kj = -1; kj <= 1; ++kj) {
                sumX += input[y + ki][x + kj] * gx[ki + 1][kj + 1];
                sumY += input[y + ki][x + kj] * gy[ki + 1][kj + 1];
            }
        }
        local_x[j] = static_cast<int>(round(sumX));
        local_y[j] = static_cast<int>(round(sumY));
        i += 1;
    }

    vector<int> gathered_x(world_size * elements_per_process);
    vector<int> gathered_y(world_size * elements_per_process);
    MPI_Allgather(local_x.data(), elements_per_process, MPI_INT, gathered_x.data(), elements_per_process, MPI_INT, MPI_COMM_WORLD);
    MPI_Allgather(local_y.data(), elements_per_process, MPI_INT, gathered_y.data(), elements_per_process, MPI_INT, MPI_COMM_WORLD);
    
    int idx = start;
    for (int j = 0; j < ((height - 2)*(width - 2)); ++j) {
        if((idx - (width - 1)) % width == 0){
            idx += 2;
        }
        int x = idx % width;
        int y = idx / width;
        gradX[y][x] = gathered_x[j];
        gradY[y][x] = gathered_y[j];
        idx += 1;
    }
    if (world_rank == 0 && DEBUG) {
        logfile << "Gradients computed:\n";
        logfile << "Gradient X:\n";
        logMatrix(gradX, logfile);
        logfile << "Gradient Y:\n";
        logMatrix(gradY, logfile);
        logfile << "\n";
    }
}

// Function to compute the magnitude and direction
void computeMagnitudeDirection(const vector<vector<int>>& gradX, const vector<vector<int>>& gradY, vector<vector<int>>& magnitude, vector<vector<int>>& direction, int width, int height, int world_size, int world_rank, ofstream& logfile) {
    int elements_per_process = ceil((height - 2)*(width - 2) / world_size);
    vector<int> local_magnitude(elements_per_process, 0);
    vector<int> local_direction(elements_per_process, 0);
    int start = width + 1;
    int offset = (world_rank * elements_per_process)/(width-2);
    int start_index = start + world_rank*elements_per_process + offset*2;

    int i = start_index;
    for(int j = 0; j < elements_per_process; ++j) {
        if((i - (width - 1)) % width == 0){
            i += 2;
        }
        int x = i % width;
        int y = i / width;
        local_magnitude[j] = (int)round(sqrt(gradX[y][x] * gradX[y][x] + gradY[y][x] * gradY[y][x]));
        local_direction[j] = (int)round(atan2(gradY[y][x], gradX[y][x]) * 180 / M_PI);
        i += 1;
    }

    vector<int> gathered_magnitude(world_size * elements_per_process);
    vector<int> gathered_direction(world_size * elements_per_process);
    MPI_Allgather(local_magnitude.data(), elements_per_process, MPI_INT, gathered_magnitude.data(), elements_per_process, MPI_INT, MPI_COMM_WORLD);
    MPI_Allgather(local_direction.data(), elements_per_process, MPI_INT, gathered_direction.data(), elements_per_process, MPI_INT, MPI_COMM_WORLD);
    
    int idx = start;
    for (int j = 0; j < ((height - 2)*(width - 2)); ++j) {
        if((idx - (width - 1)) % width == 0){
            idx += 2;
        }
        int x = idx % width;
        int y = idx / width;
        magnitude[y][x] = gathered_magnitude[j];
        direction[y][x] = gathered_direction[j];
        idx += 1;
    }

    if (world_rank == 0 && DEBUG) {
        logfile << "Magnitude and direction computed:\n";
        logfile << "Magnitude:\n";
        logMatrix(magnitude, logfile);
        logfile << "Direction:\n";
        logMatrix(direction, logfile);
        logfile << "\n";
    }
}

// Function to apply non-maximum suppression
void nonMaximumSuppression(const vector<vector<int>>& magnitude, const vector<vector<int>>& direction, vector<vector<int>>& suppressed, int width, int height, int world_size, int world_rank, ofstream& logfile) {
    int elements_per_process = ceil((height - 2)*(width - 2) / world_size);
    vector<int> local_suppressed(elements_per_process, 0);
    int start = width + 1;
    int offset = (world_rank * elements_per_process)/(width-2);
    int start_index = start + world_rank*elements_per_process + offset*2;
    
    //print start_index in cout by process
    // cout << "Process " << world_rank << " start_index: " << start_index << endl;

    int i = start_index;
    for(int j = 0; j < elements_per_process; ++j) {
        if((i - (width - 1)) % width == 0){
            i += 2;
        }
        int x = i % width;
        int y = i / width;

        int angle = direction[y][x] % 180;
        if (angle < 0) angle += 180;

        if ((angle >= 0 && angle < 22.5) || (angle >= 157.5 && angle < 180)) {
            if (magnitude[y][x] < magnitude[y][x - 1] || magnitude[y][x] < magnitude[y][x + 1]) {
                local_suppressed[j] = 0;
            } else {
                local_suppressed[j] = magnitude[y][x];
            }
        }
        else if (angle >= 22.5 && angle < 67.5) {
            if (magnitude[y][x] < magnitude[y - 1][x + 1] || magnitude[y][x] < magnitude[y + 1][x - 1]) {
                local_suppressed[j] = 0;
            } else {
                local_suppressed[j] = magnitude[y][x];
            }
        }
        else if (angle >= 67.5 && angle < 112.5) {
            if (magnitude[y][x] < magnitude[y - 1][x] || magnitude[y][x] < magnitude[y + 1][x]) {
                local_suppressed[j] = 0;
            } else {
                local_suppressed[j] = magnitude[y][x];
            }
        }
        else if (angle >= 112.5 && angle < 157.5) {
            if (magnitude[y][x] < magnitude[y - 1][x - 1] || magnitude[y][x] < magnitude[y + 1][x + 1]) {
                local_suppressed[j] = 0;
            } else {
                local_suppressed[j] = magnitude[y][x];
            }
        }
        i += 1;
    }

    vector<int> gathered_suppressed(world_size * elements_per_process);
    MPI_Allgather(local_suppressed.data(), elements_per_process, MPI_INT, gathered_suppressed.data(), elements_per_process, MPI_INT, MPI_COMM_WORLD);
    
    int idx = start;
    for (int j = 0; j < ((height - 2)*(width - 2)); ++j) {
        if((idx - (width - 1)) % width == 0){
            idx += 2;
        }
        int x = idx % width;
        int y = idx / width;
        suppressed[y][x] = gathered_suppressed[j];
        idx += 1;
    }

    if (world_rank == 0 && DEBUG) {
        logfile << "Non-maximum suppression applied:\n";
        logMatrix(suppressed, logfile);
        logfile << "\n";
    }
}

// Function to apply double thresholding
void doubleThresholding(const vector<vector<int>>& suppressed, vector<vector<int>>& edge, int width, int height, int lowThresh, int highThresh, int world_size, int world_rank, ofstream& logfile) {
    int elements_per_process = ceil((height - 2)*(width - 2) / world_size);
    vector<int> local_edge(elements_per_process, 0);
    int start = width + 1;
    int offset = (world_rank * elements_per_process)/(width-2);
    int start_index = start + world_rank*elements_per_process + offset*2;
    
    int i = start_index;
    for(int j = 0; j < elements_per_process; ++j) {
        if((i - (width - 1)) % width == 0){
            i += 2;
        }
        int x = i % width;
        int y = i / width;
        if (suppressed[y][x] >= highThresh) {
            local_edge[j] = 255;
        } else if (suppressed[y][x] >= lowThresh) {
            local_edge[j] = 128;
        } else {
            local_edge[j] = 0;
        }
        i += 1;
    }

    vector<int> gathered_edge(world_size * elements_per_process);
    MPI_Allgather(local_edge.data(), elements_per_process, MPI_INT, gathered_edge.data(), elements_per_process, MPI_INT, MPI_COMM_WORLD);
    
    int idx = start;
    for (int j = 0; j < ((height - 2)*(width - 2)); ++j) {
        if((idx - (width - 1)) % width == 0){
            idx += 2;
        }
        int x = idx % width;
        int y = idx / width;
        edge[y][x] = gathered_edge[j];
        idx += 1;
    }

    if (world_rank == 0 && DEBUG) {
        logfile << "Double thresholding applied:\n";
        logMatrix(edge, logfile);
        logfile << "\n";
        logfile.close();
    }
}

// Function to perform edge tracking by hysteresis
void edgeTrackingByHysteresis(vector<vector<int>>& edge, int width, int height) {
    for (int i = 1; i < height - 1; ++i) {
        for (int j = 1; j < width - 1; ++j) {
            if (edge[i][j] == 128) {
                if (edge[i - 1][j - 1] == 255 || edge[i - 1][j] == 255 || edge[i - 1][j + 1] == 255 ||
                    edge[i][j - 1] == 255 || edge[i][j + 1] == 255 ||
                    edge[i + 1][j - 1] == 255 || edge[i + 1][j] == 255 || edge[i + 1][j + 1] == 255) {
                    edge[i][j] = 255;
                } else {
                    edge[i][j] = 0;
                }
            }
        }
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    string filename = "input/emoji.png";
    string output_filename = "output/emoji.png";
    vector<vector<int>> image, blurredImage, gradX, gradY, magnitude, direction, suppressed, edge;
    int width, height;

    ofstream logfile;

    if(world_rank == 0 && DEBUG) {
        logfile.open("logs/mpi.txt", ios::out | ios::trunc);
        if (!logfile.is_open()) {
            cerr << "Failed to open log file.\n";
            return 1;
        }

        logfile << "==== Canny ====\n";
    }

    image = read_png(filename);
    if (image.empty()) {
        cerr << "Error reading image!" << endl;
        return -1;
    }
    width = image[0].size();
    height = image.size();
    auto start = high_resolution_clock::now();

    blurredImage = image;
    gaussianBlur(image, blurredImage, width, height, world_size, world_rank, logfile);
    
    gradX.resize(height, vector<int>(width));
    gradY.resize(height, vector<int>(width));
    computeGradients(blurredImage, gradX, gradY, width, height, world_size, world_rank, logfile);
    
    magnitude.resize(height, vector<int>(width));
    direction.resize(height, vector<int>(width));
    computeMagnitudeDirection(gradX, gradY, magnitude, direction, width, height, world_size, world_rank, logfile);
    
    suppressed.resize(height, vector<int>(width));
    nonMaximumSuppression(magnitude, direction, suppressed, width, height, world_size, world_rank, logfile);
    
    edge.resize(height, vector<int>(width));
    doubleThresholding(suppressed, edge, width, height, 30, 100, world_size, world_rank, logfile);
    
    edgeTrackingByHysteresis(edge, width, height);
    
    if (world_rank == 0) {
        auto end = high_resolution_clock::now();
        auto duration = duration_cast<milliseconds>(end - start);

        cout << "Execution time: " << duration.count() << " ms" << endl;
        write_png(output_filename, edge, width, height);
        if(DEBUG) logfile.close();
    }
    MPI_Finalize();
    return 0;
}
