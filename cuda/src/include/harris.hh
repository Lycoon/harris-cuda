#pragma once
#include <cstddef>
#include <memory>

#include "png.hh"

struct rgb_png
{
    png_byte r;
    png_byte g;
    png_byte b;
};

struct point
{
    int y;
    int x;
};

void harris(char* host_buffer, char* out_buffer, point* out_point,
            int* nb_points, size_t width, size_t height, std::ptrdiff_t stride);
