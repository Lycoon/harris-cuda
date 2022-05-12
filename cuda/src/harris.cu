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

__global__ void img2float(char* buffer, size_t pitch, float* out,
                          size_t pitch_out, size_t width, size_t height)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    rgb_png* line = (rgb_png*)(buffer + y * pitch);
    float* out_line = out + y * pitch_out;

    float r = static_cast<float>(line[x].r) * 0.299;
    float g = static_cast<float>(line[x].g) * 0.587;
    float b = static_cast<float>(line[x].b) * 0.114;

    out_line[x] = r + g + b;
}

__global__ void float2img(float* buffer, size_t pitch, char* out,
                          size_t pitch_out, size_t width, size_t height)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    float* line = buffer + y * pitch;
    rgb_png* out_line = (rgb_png*)(out + y * pitch_out);

    png_byte gray = static_cast<png_byte>(line[x]);

    out_line[x] = { gray, gray, gray };
}

void harris(char* host_buffer, size_t width, size_t height,
            std::ptrdiff_t stride)
{
    cudaError_t rc = cudaSuccess;

    char* image;
    size_t pitch_img;

    float* buffer;
    size_t pitch_buffer;

    rc = cudaMallocPitch(&image, &pitch_img, width * sizeof(rgb_png), height);
    if (rc)
        abortError("Fail buffer allocation");

    rc = cudaMallocPitch(&buffer, &pitch_buffer, width * sizeof(float), height);
    if (rc)
        abortError("Fail buffer allocation");

    rc = cudaMemcpy2D(image, pitch_img, host_buffer, stride, stride, height,
                      cudaMemcpyHostToDevice);
    if (rc)
        abortError("Unable to copy buffer from memory");

    int bsize = 32;
    int w = std::ceil((float)width / bsize);
    int h = std::ceil((float)height / bsize);

    dim3 dimBlock(bsize, bsize);
    dim3 dimGrid(w, h);

    img2float<<<dimGrid, dimBlock>>>(image, pitch_img, buffer, pitch_buffer,
                                     width, height);
    float2img<<<dimGrid, dimBlock>>>(buffer, pitch_buffer, image, pitch_img,
                                     width, height);

    if (cudaPeekAtLastError())
        abortError("Computation Error");

    rc = cudaMemcpy2D(host_buffer, stride, image, pitch_img,
                      width * sizeof(rgb_png), height, cudaMemcpyDeviceToHost);
    if (rc)
        abortError("Unable to copy buffer back to memory");

    // Free
    rc = cudaFree(image);
    if (rc)
        abortError("Unable to free memory");
}
