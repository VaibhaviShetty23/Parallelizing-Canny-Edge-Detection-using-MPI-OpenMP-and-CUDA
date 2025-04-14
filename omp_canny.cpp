#include <iostream>
#include <vector>
#define _USE_MATH_DEFINES
#include <cmath>
#include <fstream>
#include <omp.h>
#include <sstream>
#include "png_read_write.h"
#include <chrono>
using namespace std;
using namespace std::chrono;

const bool DEBUG = false; // Set to true for debugging

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
void gaussianBlur(const vector<vector<int>>& input, vector<vector<int>>& output, int width, int height, ofstream& logfile) {
    #pragma omp parallel for collapse(2)
    for (int i = 2; i < height - 2; ++i) {
        for (int j = 2; j < width - 2; ++j) {
            float sum = 0;
            for (int ki = -2; ki <= 2; ++ki) {
                for (int kj = -2; kj <= 2; ++kj) {
                    sum += input[i + ki][j + kj] * kernel[ki + 2][kj + 2];
                }
            }
            output[i][j] = static_cast<int>(round(sum));
        }
    }
    if(DEBUG){
        logfile << "Gaussian Blur applied:\n";
        logMatrix(output, logfile);
        logfile << "\n";
    }
}

// Function to compute gradients
void computeGradients(const vector<vector<int>>& input, vector<vector<int>>& gradX, vector<vector<int>>& gradY, int width, int height, ofstream& logfile) {
    int gx[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    int gy[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};

    #pragma omp parallel for collapse(2)
    for (int i = 1; i < height - 1; ++i) {
        for (int j = 1; j < width - 1; ++j) {
            int sumX = 0, sumY = 0;
            for (int ki = -1; ki <= 1; ++ki) {
                for (int kj = -1; kj <= 1; ++kj) {
                    sumX += input[i + ki][j + kj] * gx[ki + 1][kj + 1];
                    sumY += input[i + ki][j + kj] * gy[ki + 1][kj + 1];
                }
            }
            gradX[i][j] = sumX;
            gradY[i][j] = sumY;
        }
    }
    if(DEBUG){
        logfile << "Gradients computed:\n";
        logfile << "Gradient X:\n";
        logMatrix(gradX, logfile);
        logfile << "Gradient Y:\n";
        logMatrix(gradY, logfile);
        logfile << "\n";
    }
}

// Function to compute the magnitude and direction
void computeMagnitudeDirection(const vector<vector<int>>& gradX, const vector<vector<int>>& gradY, vector<vector<int>>& magnitude, vector<vector<int>>& direction, int width, int height, ofstream& logfile) {
    #pragma omp parallel for collapse(2)
    for (int i = 1; i < height - 1; ++i) {
        for (int j = 1; j < width - 1; ++j) {
            magnitude[i][j] = (int)round(sqrt(gradX[i][j] * gradX[i][j] + gradY[i][j] * gradY[i][j]));
            direction[i][j] = (int)round(atan2(gradY[i][j], gradX[i][j]) * 180 / M_PI);
        }
    }
    if(DEBUG){
        logfile << "Magnitude and direction computed:\n";
        logfile << "Magnitude:\n";
        logMatrix(magnitude, logfile);
        logfile << "Direction:\n";
        logMatrix(direction, logfile);
        logfile << "\n";
    }
}

// Function to apply non-maximum suppression
void nonMaximumSuppression(const vector<vector<int>>& magnitude, const vector<vector<int>>& direction, vector<vector<int>>& suppressed, int width, int height, ofstream& logfile) {
    #pragma omp parallel for collapse(2)
    for (int i = 1; i < height - 1; ++i) {
        for (int j = 1; j < width - 1; ++j) {
            int angle = direction[i][j] % 180;
            if (angle < 0) angle += 180;

            if ((angle >= 0 && angle < 22.5) || (angle >= 157.5 && angle < 180)) {
                if (magnitude[i][j] < magnitude[i][j - 1] || magnitude[i][j] < magnitude[i][j + 1]) {
                    suppressed[i][j] = 0;
                } else {
                    suppressed[i][j] = magnitude[i][j];
                }
            }
            else if (angle >= 22.5 && angle < 67.5) {
                if (magnitude[i][j] < magnitude[i - 1][j + 1] || magnitude[i][j] < magnitude[i + 1][j - 1]) {
                    suppressed[i][j] = 0;
                } else {
                    suppressed[i][j] = magnitude[i][j];
                }
            }
            else if (angle >= 67.5 && angle < 112.5) {
                if (magnitude[i][j] < magnitude[i - 1][j] || magnitude[i][j] < magnitude[i + 1][j]) {
                    suppressed[i][j] = 0;
                } else {
                    suppressed[i][j] = magnitude[i][j];
                }
            }
            else if (angle >= 112.5 && angle < 157.5) {
                if (magnitude[i][j] < magnitude[i - 1][j - 1] || magnitude[i][j] < magnitude[i + 1][j + 1]) {
                    suppressed[i][j] = 0;
                } else {
                    suppressed[i][j] = magnitude[i][j];
                }
            }
        }
    }
    if(DEBUG){
        logfile << "Non-maximum suppression applied:\n";
        logMatrix(suppressed, logfile);
        logfile << "\n";
    }
}

// Function to apply double thresholding
void doubleThresholding(const vector<vector<int>>& suppressed, vector<vector<int>>& edge, int width, int height, int lowThresh, int highThresh, ofstream& logfile) {
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            if (suppressed[i][j] >= highThresh) {
                edge[i][j] = 255;
            } else if (suppressed[i][j] >= lowThresh) {
                edge[i][j] = 128;
            } else {
                edge[i][j] = 0;
            }
        }
    }
    if(DEBUG){
        logfile << "Double thresholding applied:\n";
        logMatrix(edge, logfile);
        logfile << "\n";
    }
}

// Function to perform edge tracking by hysteresis
void edgeTrackingByHysteresis(vector<vector<int>>& edge, int width, int height, ofstream& logfile) {
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

int main() {
    string filename = "input/lamp.png";
    string output_filename = "output/lamp.png";
    vector<vector<int>> image, blurredImage, gradX, gradY, magnitude, direction, suppressed, edge;
    int width, height;

    ofstream logfile;
    if(DEBUG){
        logfile.open("logs/omp.txt", ios::out | ios::trunc);
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
    gaussianBlur(image, blurredImage, width, height, logfile);

    gradX.resize(height, vector<int>(width));
    gradY.resize(height, vector<int>(width));
    computeGradients(blurredImage, gradX, gradY, width, height, logfile);

    magnitude.resize(height, vector<int>(width));
    direction.resize(height, vector<int>(width));
    computeMagnitudeDirection(gradX, gradY, magnitude, direction, width, height, logfile);

    suppressed.resize(height, vector<int>(width));
    nonMaximumSuppression(magnitude, direction, suppressed, width, height, logfile);

    edge.resize(height, vector<int>(width));
    doubleThresholding(suppressed, edge, width, height, 30, 100, logfile);

    edgeTrackingByHysteresis(edge, width, height, logfile);

    auto end = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(end - start);

    cout << "Execution time: " << duration.count() << " ms" << endl;
    write_png(output_filename, edge, width, height);
    if(DEBUG) logfile.close();
    
    return 0;
}
