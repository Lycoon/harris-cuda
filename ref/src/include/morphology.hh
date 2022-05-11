#pragma once

#include <algorithm>

#include "include/matrix.hh"

Matrix* ellipse_kernel(size_t width, size_t height);
Matrix* rectangle_kernel(size_t width, size_t height);

Matrix* dilation(Matrix& matrix, Matrix& kernel);
Matrix* erosion(Matrix& matrix, Matrix& kernel);
Matrix* opening(Matrix& matrix, Matrix& kernel);
Matrix* closing(Matrix& matrix, Matrix& kernel);

Matrix* eroded_mask(Matrix& grayscale_image, size_t border);
Matrix* harris_response(Matrix& harris_img);
std::vector<Point> best_harris_points(Matrix& harris_img);
