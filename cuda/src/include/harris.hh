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

void harris(char* host_buffer, char* out_buffer, size_t width, size_t height,
            std::ptrdiff_t stride);
