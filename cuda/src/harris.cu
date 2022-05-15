#include <cassert>
#include <cstddef>
#include <iostream>
#include <memory>
#include <spdlog/spdlog.h>
#include <thrust/device_vector.h>
#include <thrust/extrema.h>

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

#define LINE(buf, n, pitch) ((float*)(buf + (n) * (pitch)))

#define NTH_BUFFER(buffers, n, pitch, padded_height)                           \
    (buffers + (n) * (padded_height) * (pitch))

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

__global__ void pad_image(char* in, size_t pitch_in, char* out,
                          size_t pitch_out, size_t width, size_t height,
                          size_t padding)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    float* line_in = LINE(in, y, pitch_in);
    float* line_out = LINE(out, y + padding, pitch_out);

    line_out[x + padding] = line_in[x];
}

__global__ void unpad_image(char* in, size_t pitch_in, char* out,
                            size_t pitch_out, size_t width, size_t height,
                            size_t padding)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    float* line_in = LINE(in, y + padding, pitch_in);
    float* line_out = LINE(out, y, pitch_out);

    line_out[x] = line_in[x + padding];
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

const size_t ELLIPSE_DIM = 20;

__device__ const uint16_t ELLIPSE_POINTS[] = {
    (0 << 8) | 10,  (1 << 8) | 6,   (1 << 8) | 7,   (1 << 8) | 8,
    (1 << 8) | 9,   (1 << 8) | 10,  (1 << 8) | 11,  (1 << 8) | 12,
    (1 << 8) | 13,  (1 << 8) | 14,  (2 << 8) | 4,   (2 << 8) | 5,
    (2 << 8) | 6,   (2 << 8) | 7,   (2 << 8) | 8,   (2 << 8) | 9,
    (2 << 8) | 10,  (2 << 8) | 11,  (2 << 8) | 12,  (2 << 8) | 13,
    (2 << 8) | 14,  (2 << 8) | 15,  (2 << 8) | 16,  (3 << 8) | 3,
    (3 << 8) | 4,   (3 << 8) | 5,   (3 << 8) | 6,   (3 << 8) | 7,
    (3 << 8) | 8,   (3 << 8) | 9,   (3 << 8) | 10,  (3 << 8) | 11,
    (3 << 8) | 12,  (3 << 8) | 13,  (3 << 8) | 14,  (3 << 8) | 15,
    (3 << 8) | 16,  (3 << 8) | 17,  (4 << 8) | 2,   (4 << 8) | 3,
    (4 << 8) | 4,   (4 << 8) | 5,   (4 << 8) | 6,   (4 << 8) | 7,
    (4 << 8) | 8,   (4 << 8) | 9,   (4 << 8) | 10,  (4 << 8) | 11,
    (4 << 8) | 12,  (4 << 8) | 13,  (4 << 8) | 14,  (4 << 8) | 15,
    (4 << 8) | 16,  (4 << 8) | 17,  (4 << 8) | 18,  (5 << 8) | 1,
    (5 << 8) | 2,   (5 << 8) | 3,   (5 << 8) | 4,   (5 << 8) | 5,
    (5 << 8) | 6,   (5 << 8) | 7,   (5 << 8) | 8,   (5 << 8) | 9,
    (5 << 8) | 10,  (5 << 8) | 11,  (5 << 8) | 12,  (5 << 8) | 13,
    (5 << 8) | 14,  (5 << 8) | 15,  (5 << 8) | 16,  (5 << 8) | 17,
    (5 << 8) | 18,  (5 << 8) | 19,  (6 << 8) | 1,   (6 << 8) | 2,
    (6 << 8) | 3,   (6 << 8) | 4,   (6 << 8) | 5,   (6 << 8) | 6,
    (6 << 8) | 7,   (6 << 8) | 8,   (6 << 8) | 9,   (6 << 8) | 10,
    (6 << 8) | 11,  (6 << 8) | 12,  (6 << 8) | 13,  (6 << 8) | 14,
    (6 << 8) | 15,  (6 << 8) | 16,  (6 << 8) | 17,  (6 << 8) | 18,
    (6 << 8) | 19,  (7 << 8) | 0,   (7 << 8) | 1,   (7 << 8) | 2,
    (7 << 8) | 3,   (7 << 8) | 4,   (7 << 8) | 5,   (7 << 8) | 6,
    (7 << 8) | 7,   (7 << 8) | 8,   (7 << 8) | 9,   (7 << 8) | 10,
    (7 << 8) | 11,  (7 << 8) | 12,  (7 << 8) | 13,  (7 << 8) | 14,
    (7 << 8) | 15,  (7 << 8) | 16,  (7 << 8) | 17,  (7 << 8) | 18,
    (7 << 8) | 19,  (8 << 8) | 0,   (8 << 8) | 1,   (8 << 8) | 2,
    (8 << 8) | 3,   (8 << 8) | 4,   (8 << 8) | 5,   (8 << 8) | 6,
    (8 << 8) | 7,   (8 << 8) | 8,   (8 << 8) | 9,   (8 << 8) | 10,
    (8 << 8) | 11,  (8 << 8) | 12,  (8 << 8) | 13,  (8 << 8) | 14,
    (8 << 8) | 15,  (8 << 8) | 16,  (8 << 8) | 17,  (8 << 8) | 18,
    (8 << 8) | 19,  (9 << 8) | 0,   (9 << 8) | 1,   (9 << 8) | 2,
    (9 << 8) | 3,   (9 << 8) | 4,   (9 << 8) | 5,   (9 << 8) | 6,
    (9 << 8) | 7,   (9 << 8) | 8,   (9 << 8) | 9,   (9 << 8) | 10,
    (9 << 8) | 11,  (9 << 8) | 12,  (9 << 8) | 13,  (9 << 8) | 14,
    (9 << 8) | 15,  (9 << 8) | 16,  (9 << 8) | 17,  (9 << 8) | 18,
    (9 << 8) | 19,  (10 << 8) | 0,  (10 << 8) | 1,  (10 << 8) | 2,
    (10 << 8) | 3,  (10 << 8) | 4,  (10 << 8) | 5,  (10 << 8) | 6,
    (10 << 8) | 7,  (10 << 8) | 8,  (10 << 8) | 9,  (10 << 8) | 10,
    (10 << 8) | 11, (10 << 8) | 12, (10 << 8) | 13, (10 << 8) | 14,
    (10 << 8) | 15, (10 << 8) | 16, (10 << 8) | 17, (10 << 8) | 18,
    (10 << 8) | 19, (11 << 8) | 0,  (11 << 8) | 1,  (11 << 8) | 2,
    (11 << 8) | 3,  (11 << 8) | 4,  (11 << 8) | 5,  (11 << 8) | 6,
    (11 << 8) | 7,  (11 << 8) | 8,  (11 << 8) | 9,  (11 << 8) | 10,
    (11 << 8) | 11, (11 << 8) | 12, (11 << 8) | 13, (11 << 8) | 14,
    (11 << 8) | 15, (11 << 8) | 16, (11 << 8) | 17, (11 << 8) | 18,
    (11 << 8) | 19, (12 << 8) | 0,  (12 << 8) | 1,  (12 << 8) | 2,
    (12 << 8) | 3,  (12 << 8) | 4,  (12 << 8) | 5,  (12 << 8) | 6,
    (12 << 8) | 7,  (12 << 8) | 8,  (12 << 8) | 9,  (12 << 8) | 10,
    (12 << 8) | 11, (12 << 8) | 12, (12 << 8) | 13, (12 << 8) | 14,
    (12 << 8) | 15, (12 << 8) | 16, (12 << 8) | 17, (12 << 8) | 18,
    (12 << 8) | 19, (13 << 8) | 0,  (13 << 8) | 1,  (13 << 8) | 2,
    (13 << 8) | 3,  (13 << 8) | 4,  (13 << 8) | 5,  (13 << 8) | 6,
    (13 << 8) | 7,  (13 << 8) | 8,  (13 << 8) | 9,  (13 << 8) | 10,
    (13 << 8) | 11, (13 << 8) | 12, (13 << 8) | 13, (13 << 8) | 14,
    (13 << 8) | 15, (13 << 8) | 16, (13 << 8) | 17, (13 << 8) | 18,
    (13 << 8) | 19, (14 << 8) | 1,  (14 << 8) | 2,  (14 << 8) | 3,
    (14 << 8) | 4,  (14 << 8) | 5,  (14 << 8) | 6,  (14 << 8) | 7,
    (14 << 8) | 8,  (14 << 8) | 9,  (14 << 8) | 10, (14 << 8) | 11,
    (14 << 8) | 12, (14 << 8) | 13, (14 << 8) | 14, (14 << 8) | 15,
    (14 << 8) | 16, (14 << 8) | 17, (14 << 8) | 18, (14 << 8) | 19,
    (15 << 8) | 1,  (15 << 8) | 2,  (15 << 8) | 3,  (15 << 8) | 4,
    (15 << 8) | 5,  (15 << 8) | 6,  (15 << 8) | 7,  (15 << 8) | 8,
    (15 << 8) | 9,  (15 << 8) | 10, (15 << 8) | 11, (15 << 8) | 12,
    (15 << 8) | 13, (15 << 8) | 14, (15 << 8) | 15, (15 << 8) | 16,
    (15 << 8) | 17, (15 << 8) | 18, (15 << 8) | 19, (16 << 8) | 2,
    (16 << 8) | 3,  (16 << 8) | 4,  (16 << 8) | 5,  (16 << 8) | 6,
    (16 << 8) | 7,  (16 << 8) | 8,  (16 << 8) | 9,  (16 << 8) | 10,
    (16 << 8) | 11, (16 << 8) | 12, (16 << 8) | 13, (16 << 8) | 14,
    (16 << 8) | 15, (16 << 8) | 16, (16 << 8) | 17, (16 << 8) | 18,
    (17 << 8) | 3,  (17 << 8) | 4,  (17 << 8) | 5,  (17 << 8) | 6,
    (17 << 8) | 7,  (17 << 8) | 8,  (17 << 8) | 9,  (17 << 8) | 10,
    (17 << 8) | 11, (17 << 8) | 12, (17 << 8) | 13, (17 << 8) | 14,
    (17 << 8) | 15, (17 << 8) | 16, (17 << 8) | 17, (18 << 8) | 4,
    (18 << 8) | 5,  (18 << 8) | 6,  (18 << 8) | 7,  (18 << 8) | 8,
    (18 << 8) | 9,  (18 << 8) | 10, (18 << 8) | 11, (18 << 8) | 12,
    (18 << 8) | 13, (18 << 8) | 14, (18 << 8) | 15, (18 << 8) | 16,
    (19 << 8) | 6,  (19 << 8) | 7,  (19 << 8) | 8,  (19 << 8) | 9,
    (19 << 8) | 10, (19 << 8) | 11, (19 << 8) | 12, (19 << 8) | 13,
    (19 << 8) | 14
};

#define CONVOLVE(out, in, i, j, width, height, pitch, kernel, kernel_size)     \
    {                                                                          \
        float* line = LINE(out, i, pitch);                                     \
                                                                               \
        float acc = 0;                                                         \
        size_t kI = kernel_size - 1;                                           \
                                                                               \
        int maxY = ((int)kernel_size) / 2 + kernel_size % 2;                   \
        for (int kY = -((int)kernel_size) / 2; kY < maxY; kY++, kI--)          \
        {                                                                      \
            size_t kJ = kernel_size - 1;                                       \
            int maxX = ((int)kernel_size) / 2 + kernel_size % 2;               \
            for (int kX = -((int)kernel_size) / 2; kX < maxX; kX++, kJ--)      \
            {                                                                  \
                float* current_line = (float*)(in + (i + kY) * pitch);         \
                acc += current_line[j + kX] * kernel[kI * kernel_size + kJ];   \
            }                                                                  \
        }                                                                      \
                                                                               \
        line[j] = acc;                                                         \
    }

#define CONVOLVE_DILATE(out, in, i, j, width, height, pitch)                   \
    {                                                                          \
        float acc = 0;                                                         \
        float* line = LINE(out, i, pitch);                                     \
                                                                               \
        int offSet = 10;                                                       \
        for (uint16_t p : ELLIPSE_POINTS)                                      \
        {                                                                      \
            int kX = (int)(p & 0xFF) - offSet;                                 \
            int kY = (int)((p >> 8) & 0xFF) - offSet;                          \
                                                                               \
            float* current_line = (float*)(in + (i + kY) * pitch);             \
            acc = max(current_line[j + kX], acc);                              \
        }                                                                      \
                                                                               \
        line[j] = acc;                                                         \
    }

__global__ void gauss_derivatives(char* buffers, size_t pitch, size_t width,
                                  size_t height, size_t padding)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    x += padding;
    y += padding;

    char* im = NTH_BUFFER(buffers, 0, pitch, height + (2 * padding));
    char* im_x = NTH_BUFFER(buffers, 1, pitch, height + (2 * padding));
    char* im_y = NTH_BUFFER(buffers, 2, pitch, height + (2 * padding));

    CONVOLVE(im_x, im, y, x, width, height, pitch, GAUSS_X, GAUSS_KERNEL_DIM);
    CONVOLVE(im_y, im, y, x, width, height, pitch, GAUSS_Y, GAUSS_KERNEL_DIM);

    char* im_xx = NTH_BUFFER(buffers, 3, pitch, height + (2 * padding));
    char* im_xy = NTH_BUFFER(buffers, 4, pitch, height + (2 * padding));
    char* im_yy = NTH_BUFFER(buffers, 5, pitch, height + (2 * padding));

    float* line_im_x = LINE(im_x, y, pitch);
    float* line_im_y = LINE(im_y, y, pitch);
    float* line_im_xx = LINE(im_xx, y, pitch);
    float* line_im_xy = LINE(im_xy, y, pitch);
    float* line_im_yy = LINE(im_yy, y, pitch);

    line_im_xx[x] = line_im_x[x] * line_im_x[x];
    line_im_xy[x] = line_im_x[x] * line_im_y[x];
    line_im_yy[x] = line_im_y[x] * line_im_y[x];
}

__global__ void harris_img(char* buffers, size_t pitch, size_t width,
                           size_t height, size_t padding)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    x += padding;
    y += padding;

    char* im_xx = NTH_BUFFER(buffers, 3, pitch, height + (2 * padding));
    char* im_xy = NTH_BUFFER(buffers, 4, pitch, height + (2 * padding));
    char* im_yy = NTH_BUFFER(buffers, 5, pitch, height + (2 * padding));

    char* W_xx = NTH_BUFFER(buffers, 0, pitch, height + (2 * padding));
    char* W_xy = NTH_BUFFER(buffers, 1, pitch, height + (2 * padding));
    char* W_yy = NTH_BUFFER(buffers, 2, pitch, height + (2 * padding));

    CONVOLVE(W_xx, im_xx, y, x, width, height, pitch, GAUSS_KERNEL,
             GAUSS_KERNEL_DIM);
    CONVOLVE(W_xy, im_xy, y, x, width, height, pitch, GAUSS_KERNEL,
             GAUSS_KERNEL_DIM);
    CONVOLVE(W_yy, im_yy, y, x, width, height, pitch, GAUSS_KERNEL,
             GAUSS_KERNEL_DIM);

    float* line_W_xx = LINE(W_xx, y, pitch);
    float* line_W_xy = LINE(W_xy, y, pitch);
    float* line_W_yy = LINE(W_yy, y, pitch);

    char* W_xy_2 = NTH_BUFFER(buffers, 6, pitch, height + (2 * padding));
    float* line_W_xy_2 = LINE(W_xy_2, y, pitch);
    line_W_xy_2[x] = line_W_xy[x] * line_W_xy[x];

    char* W_tr = NTH_BUFFER(buffers, 7, pitch, height + (2 * padding));
    float* line_W_tr = LINE(W_tr, y, pitch);
    line_W_tr[x] = line_W_xx[x] + line_W_yy[x] + 1;

    char* W_det = NTH_BUFFER(buffers, 8, pitch, height + (2 * padding));
    float* line_W_det = LINE(W_det, y, pitch);
    line_W_det[x] = line_W_xx[x] * line_W_yy[x] - line_W_xy_2[x];

    line_W_det[x] = line_W_det[x] / line_W_tr[x];
}

__global__ void threshold(char* buffer, size_t pitch, size_t width,
                          size_t height, float threshold, size_t padding)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    x += padding;
    y += padding;

    float* line_buffer = LINE(buffer, y, pitch);
    line_buffer[x] = line_buffer[x] > threshold ? 1.0 : 0.0;
}

__global__ void dilate(char* out, char* in, size_t pitch, size_t width,
                       size_t height, size_t padding)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    x += padding;
    y += padding;

    CONVOLVE_DILATE(out, in, y, x, width, height, pitch);
}

__global__ void harris_response(char* harris_im, char* harris_dil, size_t pitch,
                                size_t width, size_t height, float min,
                                float max, size_t padding)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    x += padding;
    y += padding;

    float* line_harris_im = LINE(harris_im, y, pitch);
    float* line_harris_dil = LINE(harris_dil, y, pitch);

    int is_close = abs(line_harris_im[x] - line_harris_dil[x])
        <= (1.0e-8 + 1.0e-5 * abs(line_harris_dil[x]));

    line_harris_dil[x] =
        line_harris_im[x] > (min + 0.5 * (max - min)) ? line_harris_im[x] : 0;
    line_harris_dil[x] = line_harris_dil[x] * (float)is_close;
}

__global__ void best_points(char* harris_resp, point* points, char* values,
                            size_t pitch, size_t width, size_t height,
                            int* count, size_t padding)
{
    int current_count = 0;

    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    x += padding;
    y += padding;

    float* line_harris_resp = LINE(harris_resp, y, pitch);
    float* line_values = ((float*)values) + y * width;
    point* line_points = points + (y * width);

    line_points[x] = { y, x };
    line_values[x] = line_harris_resp[x];

    current_count = line_harris_resp[x] >= 1e-3;
    atomicAdd(count, current_count);
}

void harris(char* host_buffer, char* out_buffer, point* out_point,
            int* nb_points, size_t width, size_t height, std::ptrdiff_t stride)
{
    cudaError_t rc = cudaSuccess;

    char* image;
    size_t pitch_img;

    char* buffer;
    size_t pitch_buffer;

    char* padded_buffers;
    size_t padding = ELLIPSE_DIM / 2 + ELLIPSE_DIM % 2;
    size_t pitch_padded_buffers;

    size_t padded_width = width + padding * 2;
    size_t padded_height = height + padding * 2;

    point* points;

    rc = cudaMallocPitch(&image, &pitch_img, width * sizeof(rgb_png), height);
    if (rc)
        abortError("Fail buffer allocation");

    rc = cudaMallocPitch(&buffer, &pitch_buffer, width * sizeof(float), height);
    if (rc)
        abortError("Fail buffer allocation");

    const size_t NB_BUFFERS = 9;

    rc = cudaMallocPitch(&padded_buffers, &pitch_padded_buffers,
                         padded_width * sizeof(float),
                         NB_BUFFERS * padded_height);
    if (rc)
        abortError("Fail buffer allocation");

    rc = cudaMemset(padded_buffers, 0,
                    NB_BUFFERS * padded_width * sizeof(float) * padded_height);
    if (rc)
        abortError("Fail buffer memset");

    rc = cudaMalloc(&points, height * width * sizeof(point));
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

    if (cudaPeekAtLastError())
        abortError("Computation Error");

    char* padded_img =
        NTH_BUFFER(padded_buffers, 0, pitch_padded_buffers, padded_height);

    pad_image<<<dimGrid, dimBlock>>>(buffer, pitch_buffer, padded_img,
                                     pitch_padded_buffers, width, height,
                                     padding);

    if (cudaPeekAtLastError())
        abortError("Computation Error");

    // [im_x, im_y, im_xx, im_xy, im_yy]
    gauss_derivatives<<<dimGrid, dimBlock>>>(
        padded_buffers, pitch_padded_buffers, width, height, padding);

    if (cudaPeekAtLastError())
        abortError("Computation Error");

    // [W_xx, W_xy, W_yy, W_xy_2, W_tr, W_det]
    harris_img<<<dimGrid, dimBlock>>>(padded_buffers, pitch_padded_buffers,
                                      width, height, padding);

    if (cudaPeekAtLastError())
        abortError("Computation Error");

    char* harris_im =
        NTH_BUFFER(padded_buffers, 8, pitch_padded_buffers, padded_height);

    thrust::device_vector<float> vec(
        (float*)harris_im,
        (float*)(harris_im + padded_height * pitch_padded_buffers));

    // TODO: use minmax
    float harris_im_min = *thrust::min_element(vec.begin(), vec.end());
    float harris_im_max = *thrust::max_element(vec.begin(), vec.end());

    if (cudaPeekAtLastError())
        abortError("Computation Error");

    char* harris_dil =
        NTH_BUFFER(padded_buffers, 1, pitch_padded_buffers, padded_height);

    rc = cudaMemcpy2D(harris_dil, pitch_padded_buffers, harris_im,
                      pitch_padded_buffers, padded_width * sizeof(float),
                      padded_height, cudaMemcpyDeviceToDevice);
    if (rc)
        abortError("Unable to copy buffer back to memory");

    dilate<<<dimGrid, dimBlock>>>(harris_dil, harris_im, pitch_padded_buffers,
                                  width, height, padding);

    if (cudaPeekAtLastError())
        abortError("Computation Error");

    harris_response<<<dimGrid, dimBlock>>>(
        harris_im, harris_dil, pitch_padded_buffers, width, height,
        harris_im_min, harris_im_max, padding);

    if (cudaPeekAtLastError())
        abortError("Computation Error");

    int* count;
    rc = cudaMalloc(&count, 1 * sizeof(int));
    if (rc)
        abortError("Fail buffer allocation");

    rc = cudaMemset(count, 0, 1 * sizeof(int));
    if (rc)
        abortError("Fail buffer memset");

    best_points<<<dimGrid, dimBlock>>>(harris_dil, points, padded_buffers,
                                       pitch_padded_buffers, width, height,
                                       count, padding);

    if (cudaPeekAtLastError())
        abortError("Computation Error");

    thrust::sort_by_key(
        thrust::device, (float*)padded_buffers,
        (float*)(padded_buffers + height * width * sizeof(float)), points);

    char* result = harris_dil;

    unpad_image<<<dimGrid, dimBlock>>>(result, pitch_padded_buffers, buffer,
                                       pitch_buffer, width, height, padding);

    if (cudaPeekAtLastError())
        abortError("Computation Error");

    rc = cudaMemcpy2D(out_buffer, width * sizeof(float), buffer, pitch_buffer,
                      width * sizeof(float), height, cudaMemcpyDeviceToHost);
    if (rc)
        abortError("Unable to copy buffer back to memory");

    rc = cudaMemcpy(nb_points, count, 1 * sizeof(int), cudaMemcpyDeviceToHost);
    if (rc)
        abortError("Unable to copy buffer back to memory");

    *nb_points = *nb_points > 2000 ? 2000 : *nb_points;
    rc = cudaMemcpy(out_point, points + width * height - *nb_points,
                    *nb_points * sizeof(point), cudaMemcpyDeviceToHost);

    if (rc)
        abortError("Unable to copy buffer back to memory");

    // Free
    rc = cudaFree(image);
    if (rc)
        abortError("Unable to free memory");

    rc = cudaFree(padded_buffers);
    if (rc)
        abortError("Unable to free memory");
}
