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

__global__ void grayscale(rgb8_t* buffer, rgb8_t* out_buf, size_t width,
                          size_t height, size_t pitch_buf, size_t pitch_out_buf)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    rgb8_t* line = buffer + y * pitch_buf;
    rgb8_t* out_line = out_buf + y * pitch_out_buf;

    float r = static_cast<float>(line[x].r) * 0.299;
    float g = static_cast<float>(line[x].g) * 0.587;
    float b = static_cast<float>(line[x].b) * 0.114;

    rgb8_t gray = { static_cast<uint8_t>(r), static_cast<uint8_t>(g),
                    static_cast<uint8_t>(b) };
    out_line[x] = gray;
}

void harris(char* host_buffer, size_t width, size_t height,
            std::ptrdiff_t stride)
{
    cudaError_t rc = cudaSuccess;

    rgb8_t* gray;
    size_t pitch;

    rc = cudaMallocPitch(&gray, &pitch, width * sizeof(rgb8_t), height);
    if (rc)
        abortError("Fail buffer allocation");

    int bsize = 32;
    int w = std::ceil((float)width / bsize);
    int h = std::ceil((float)height / bsize);

    dim3 dimBlock(bsize, bsize);
    dim3 dimGrid(w, h);

    grayscale<<<dimGrid, dimBlock>>>((rgb8_t*)host_buffer, gray, width, height,
                                     stride, pitch);

    if (cudaPeekAtLastError())
        abortError("Computation Error");

    rc = cudaMemcpy2D(host_buffer, stride, gray, pitch, width * sizeof(rgb8_t),
                      height, cudaMemcpyDeviceToHost);
    if (rc)
        abortError("Unable to copy buffer back to memory");

    // Free
    rc = cudaFree(gray);
    if (rc)
        abortError("Unable to free memory");
}
