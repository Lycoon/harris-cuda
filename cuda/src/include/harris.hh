#pragma once
#include <cstddef>
#include <memory>

struct rgb8_t
{
    std::uint8_t r;
    std::uint8_t g;
    std::uint8_t b;
};

void harris(char* host_buffer, size_t width, size_t height,
            std::ptrdiff_t stride);
