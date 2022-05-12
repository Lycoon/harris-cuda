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

#define FLOAT_LINE(buf, y) ((float*)(((char*)buf) + y))

__global__ void img2float(char* buffer, size_t pitch, char* out,
                          size_t pitch_out, size_t width, size_t height)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    rgb_png* line = (rgb_png*)(buffer + y * pitch);
    float* out_line = (float*)(out + y * pitch_out);

    float r = static_cast<float>(line[x].r) * 0.299;
    float g = static_cast<float>(line[x].g) * 0.587;
    float b = static_cast<float>(line[x].b) * 0.114;

    out_line[x] = r + g + b;
}

const size_t GAUSS_KERNEL_DIM = 7;

__device__ const float GAUSS_X[] = {
    0.000308397,  0.00263512,  0.00608749, -0,         -0.00608749, -0.00263512,
    -0.000308397, 0.00395267,  0.0337738,  0.0780224,  -0,          -0.0780224,
    -0.0337738,   -0.00395267, 0.0182625,  0.156045,   0.360485,    -0,
    -0.360485,    -0.156045,   -0.0182625, 0.0304169,  0.259899,    0.600404,
    -0,           -0.600404,   -0.259899,  -0.0304169, 0.0182625,   0.156045,
    0.360485,     -0,          -0.360485,  -0.156045,  -0.0182625,  0.00395267,
    0.0337738,    0.0780224,   -0,         -0.0780224, -0.0337738,  -0.00395267,
    0.000308397,  0.00263512,  0.00608749, -0,         -0.00608749, -0.00263512,
    -0.000308397,
};

__device__ const float GAUSS_Y[] = {
    0.000308397,  0.00395267,  0.0182625,  0.0304169,  0.0182625,   0.00395267,
    0.000308397,  0.00263512,  0.0337738,  0.156045,   0.259899,    0.156045,
    0.0337738,    0.00263512,  0.00608749, 0.0780224,  0.360485,    0.600404,
    0.360485,     0.0780224,   0.00608749, -0,         -0,          -0,
    -0,           -0,          -0,         -0,         -0.00608749, -0.0780224,
    -0.360485,    -0.600404,   -0.360485,  -0.0780224, -0.00608749, -0.00263512,
    -0.0337738,   -0.156045,   -0.259899,  -0.156045,  -0.0337738,  -0.00263512,
    -0.000308397, -0.00395267, -0.0182625, -0.0304169, -0.0182625,  -0.00395267,
    -0.000308397,
};

__device__ const float GAUSS_KERNEL[] = {
    0.000102799, 0.00131756, 0.00608749, 0.010139,  0.00608749, 0.00131756,
    0.000102799, 0.00131756, 0.0168869,  0.0780224, 0.12995,    0.0780224,
    0.0168869,   0.00131756, 0.00608749, 0.0780224, 0.360485,   0.600404,
    0.360485,    0.0780224,  0.00608749, 0.010139,  0.12995,    0.600404,
    1,           0.600404,   0.12995,    0.010139,  0.00608749, 0.0780224,
    0.360485,    0.600404,   0.360485,   0.0780224, 0.00608749, 0.00131756,
    0.0168869,   0.0780224,  0.12995,    0.0780224, 0.0168869,  0.00131756,
    0.000102799, 0.00131756, 0.00608749, 0.010139,  0.00608749, 0.00131756,
    0.000102799,
};

__device__ void convolve(char* out, char* in, size_t i, size_t j, size_t width,
                         size_t height, size_t pitch, const float* kernel,
                         size_t kernel_size)
{
    long k_size = static_cast<long>(kernel_size);

    float* line = (float*)(out + i * pitch);

    float acc = 0;
    size_t kI = kernel_size - 1;

    int maxY = ((int)kernel_size) / 2 + kernel_size % 2;
    for (int kY = -((int)kernel_size) / 2; kY < maxY; kY++, kI--)
    {
        size_t kJ = kernel_size - 1;
        int maxX = ((int)kernel_size) / 2 + kernel_size % 2;
        for (int kX = -((int)kernel_size) / 2; kX < maxX; kX++, kJ--)
        {
            if (((int)i) + kY >= 0 && i + kY < height && ((int)j) + kX >= 0
                && j + kX < width)
            {
                float* current_line = (float*)(in + (i + kY) * pitch);
                acc += current_line[j + kX] * kernel[kI * kernel_size + kJ];
            }
        }
    }

    line[j] = acc;
}

__global__ void gauss_derivatives(char* buffer, size_t pitch, size_t width,
                                  size_t height, char* buffers,
                                  size_t pitch_buffers)
{
    const size_t IM_X = 0;
    const size_t IM_Y = 1;

    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    char* im_x = (buffers + IM_X * height * pitch_buffers);
    char* im_y = (buffers + IM_Y * height * pitch_buffers);

    convolve(im_x, buffer, y, x, width, height, pitch_buffers, GAUSS_X,
             GAUSS_KERNEL_DIM);
    convolve(im_y, buffer, y, x, width, height, pitch_buffers, GAUSS_Y,
             GAUSS_KERNEL_DIM);
}

void harris(char* host_buffer, char* out_buffer, size_t width, size_t height,
            std::ptrdiff_t stride)
{
    cudaError_t rc = cudaSuccess;

    char* image;
    size_t pitch_img;

    char* buffer;
    size_t pitch_buffer;

    char* harris_buffers;
    size_t pitch_harris_buffers;

    rc = cudaMallocPitch(&image, &pitch_img, width * sizeof(rgb_png), height);
    if (rc)
        abortError("Fail buffer allocation");

    rc = cudaMallocPitch(&buffer, &pitch_buffer, width * sizeof(float), height);
    if (rc)
        abortError("Fail buffer allocation");

    rc = cudaMallocPitch(&harris_buffers, &pitch_harris_buffers,
                         width * sizeof(float), 2 * height);
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
    gauss_derivatives<<<dimGrid, dimBlock>>>(buffer, pitch_buffer, width,
                                             height, harris_buffers,
                                             pitch_harris_buffers);

    if (cudaPeekAtLastError())
        abortError("Computation Error");

    auto im_y = harris_buffers + 1 * height * pitch_harris_buffers;
    rc = cudaMemcpy2D(out_buffer, width * sizeof(float), im_y,
                      pitch_harris_buffers, width * sizeof(float), height,
                      cudaMemcpyDeviceToHost);
    if (rc)
        abortError("Unable to copy buffer back to memory");

    // Free
    rc = cudaFree(image);
    if (rc)
        abortError("Unable to free memory");

    rc = cudaFree(buffer);
    if (rc)
        abortError("Unable to free memory");

    rc = cudaFree(harris_buffers);
    if (rc)
        abortError("Unable to free memory");
}
