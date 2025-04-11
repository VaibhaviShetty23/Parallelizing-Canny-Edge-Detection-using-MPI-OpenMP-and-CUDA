#include "png_read_write.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <iostream>
#include <vector>
#include <string>

std::vector<std::vector<int>> read_png(const std::string& filename) {
    int width, height, channels;
    
    unsigned char* data = stbi_load(filename.c_str(), &width, &height, &channels, 3);
    
    if (!data) {
        std::cerr << "Failed to load image: " << filename << std::endl;
        return {};
    }

    std::vector<std::vector<int>> image(height, std::vector<int>(width));

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int idx = (y * width + x) * 3;
            image[y][x] = (data[idx] + data[idx + 1] + data[idx + 2]) / 3;
        }
    }

    stbi_image_free(data);
    return image;
}

void write_png(const std::string& filename, const std::vector<std::vector<int>>& image, int width, int height) {
    unsigned char* img_data = new unsigned char[height * width];
    
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int idx = y * width + x;
            img_data[idx] = static_cast<unsigned char>(image[y][x]);
        }
    }

    if (stbi_write_png(filename.c_str(), width, height, 1, img_data, width)) {
        std::cout << "Image saved successfully: " << filename << std::endl;
    } else {
        std::cerr << "Error saving the image: " << filename << std::endl;
    }

    delete[] img_data;
}
