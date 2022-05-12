#include <cassert>
#include <cstddef>
#include <iostream>
#include <memory>
#include <spdlog/spdlog.h>

#include "include/harris.hh"

[[gnu::noinline]] void _abortError(const char* msg, const char* fname, int line)
{
    cudaError_t err = cudaGetLastError();
    spdlog::error("{} ({}, line: {})", msg, fname, line);
    spdlog::error("Error {}: {}", cudaGetErrorName(err),
                  cudaGetErrorString(err));
    std::exit(1);
}

#define abortError(msg) _abortError(msg, __FUNCTION__, __LINE__)

__global__ void grayscale_img(char* buffer, size_t width, size_t height,
                              size_t pitch)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    rgb_png* line = (rgb_png*)(buffer + y * pitch);

    float r = static_cast<float>(line[x].r) * 0.299;
    float g = static_cast<float>(line[x].g) * 0.587;
    float b = static_cast<float>(line[x].b) * 0.114;

    png_byte gray = static_cast<uint8_t>(r + g + b);
    line[x] = { gray, gray, gray };
}

void harris(char* host_buffer, size_t width, size_t height,
            std::ptrdiff_t stride)
{
    cudaError_t rc = cudaSuccess;

    char* gray;
    size_t pitch;

    rc = cudaMallocPitch(&gray, &pitch, width * sizeof(rgb_png), height);
    if (rc)
        abortError("Fail buffer allocation");

    rc = cudaMemcpy2D(gray, pitch, host_buffer, stride, stride, height,
                      cudaMemcpyHostToDevice);
    if (rc)
        abortError("Unable to copy buffer from memory");

    int bsize = 32;
    int w = std::ceil((float)width / bsize);
    int h = std::ceil((float)height / bsize);

    dim3 dimBlock(bsize, bsize);
    dim3 dimGrid(w, h);

    grayscale_img<<<dimGrid, dimBlock>>>(gray, width, height, pitch);

    if (cudaPeekAtLastError())
        abortError("Computation Error");

    rc = cudaMemcpy2D(host_buffer, stride, gray, pitch, width * sizeof(rgb_png),
                      height, cudaMemcpyDeviceToHost);
    if (rc)
        abortError("Unable to copy buffer back to memory");

    // Free
    rc = cudaFree(gray);
    if (rc)
        abortError("Unable to free memory");
}
