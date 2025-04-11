#ifndef PNG_READ_WRITE_H
#define PNG_READ_WRITE_H

#include <vector>
#include <string>

std::vector<std::vector<int>> read_png(const std::string& filename);

void write_png(const std::string& filename, const std::vector<std::vector<int>>& image, int width, int height);

#endif
