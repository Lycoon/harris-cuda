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

const size_t ELLIPSE_DIM = 20;

__device__ const float ELLIPSE[] = {
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
    0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0
};

#define CONVOLVE(out, in, i, j, width, height, pitch, kernel, kernel_size)     \
    {                                                                          \
        float* line = (float*)(out + i * pitch);                               \
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
                if (((int)i) + kY >= 0 && i + kY < height                      \
                    && ((int)j) + kX >= 0 && j + kX < width)                   \
                {                                                              \
                    float* current_line = (float*)(in + (i + kY) * pitch);     \
                    acc +=                                                     \
                        current_line[j + kX] * kernel[kI * kernel_size + kJ];  \
                }                                                              \
            }                                                                  \
        }                                                                      \
                                                                               \
        line[j] = acc;                                                         \
    }

#define CONVOLVE_DILATE(out, in, i, j, width, height, pitch, kernel,           \
                        kernel_size)                                           \
    {                                                                          \
        float* line = (float*)(out + i * pitch);                               \
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
                if (((int)i) + kY >= 0 && i + kY < height                      \
                    && ((int)j) + kX >= 0 && j + kX < width)                   \
                {                                                              \
                    float* current_line = (float*)(in + (i + kY) * pitch);     \
                    if (kernel[kI * kernel_size + kJ] > 0.00001                \
                        && current_line[j + kX] > acc)                         \
                        acc = current_line[j + kX];                            \
                }                                                              \
            }                                                                  \
        }                                                                      \
                                                                               \
        line[j] = acc;                                                         \
    }

__device__ void convolve_dilate(char* out, char* in, size_t i, size_t j,
                                size_t width, size_t height, size_t pitch,
                                const float* kernel, size_t kernel_size)
{
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
                if (kernel[kI * kernel_size + kJ] > 0.00001
                    && current_line[j + kX] > acc)
                    acc = current_line[j + kX];
            }
        }
    }

    line[j] = acc;
}

__device__ char* nth_buffer(char* buffers, size_t n, size_t pitch,
                            size_t height)
{
    return buffers + n * height * pitch;
}

__device__ float* line(char* buf, size_t n, size_t pitch)
{
    return (float*)(buf + n * pitch);
}

__global__ void gauss_derivatives(char* buffer, size_t pitch, size_t width,
                                  size_t height, char* buffers)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    char* im_x = nth_buffer(buffers, 0, pitch, height);
    char* im_y = nth_buffer(buffers, 1, pitch, height);

    CONVOLVE(im_x, buffer, y, x, width, height, pitch, GAUSS_X,
             GAUSS_KERNEL_DIM);
    CONVOLVE(im_y, buffer, y, x, width, height, pitch, GAUSS_Y,
             GAUSS_KERNEL_DIM);

    char* im_xx = nth_buffer(buffers, 2, pitch, height);
    char* im_xy = nth_buffer(buffers, 3, pitch, height);
    char* im_yy = nth_buffer(buffers, 4, pitch, height);

    float* line_im_x = line(im_x, y, pitch);
    float* line_im_y = line(im_y, y, pitch);
    float* line_im_xx = line(im_xx, y, pitch);
    float* line_im_xy = line(im_xy, y, pitch);
    float* line_im_yy = line(im_yy, y, pitch);

    line_im_xx[x] = line_im_x[x] * line_im_x[x];
    line_im_xy[x] = line_im_x[x] * line_im_y[x];
    line_im_yy[x] = line_im_y[x] * line_im_y[x];
}

__global__ void harris_img(char* buffers, size_t pitch, size_t width,
                           size_t height)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    char* im_xx = nth_buffer(buffers, 2, pitch, height);
    char* im_xy = nth_buffer(buffers, 3, pitch, height);
    char* im_yy = nth_buffer(buffers, 4, pitch, height);

    char* W_xx = nth_buffer(buffers, 5, pitch, height);
    char* W_xy = nth_buffer(buffers, 6, pitch, height);
    char* W_yy = nth_buffer(buffers, 7, pitch, height);

    CONVOLVE(W_xx, im_xx, y, x, width, height, pitch, GAUSS_KERNEL,
             GAUSS_KERNEL_DIM);
    CONVOLVE(W_xy, im_xy, y, x, width, height, pitch, GAUSS_KERNEL,
             GAUSS_KERNEL_DIM);
    CONVOLVE(W_yy, im_yy, y, x, width, height, pitch, GAUSS_KERNEL,
             GAUSS_KERNEL_DIM);

    float* line_W_xx = line(W_xx, y, pitch);
    float* line_W_xy = line(W_xy, y, pitch);
    float* line_W_yy = line(W_yy, y, pitch);

    char* W_xy_2 = nth_buffer(buffers, 8, pitch, height);
    float* line_W_xy_2 = line(W_xy_2, y, pitch);
    line_W_xy_2[x] = line_W_xy[x] * line_W_xy[x];

    char* W_tr = nth_buffer(buffers, 9, pitch, height);
    float* line_W_tr = line(W_tr, y, pitch);
    line_W_tr[x] = line_W_xx[x] + line_W_yy[x] + 1;

    char* W_det = nth_buffer(buffers, 10, pitch, height);
    float* line_W_det = line(W_det, y, pitch);
    line_W_det[x] = line_W_xx[x] * line_W_yy[x] - line_W_xy_2[x];

    line_W_det[x] = line_W_det[x] / line_W_tr[x];
}

__global__ void threshold(char* buffer, size_t pitch, size_t width,
                          size_t height, float threshold)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    float* line_buffer = line(buffer, y, pitch);
    line_buffer[x] = line_buffer[x] > threshold ? 1.0 : 0.0;
}

__global__ void dilate(char* out, char* in, size_t pitch, size_t width,
                       size_t height)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    CONVOLVE_DILATE(out, in, y, x, width, height, pitch, ELLIPSE, ELLIPSE_DIM);
}

__global__ void harris_response(char* harris_im, char* harris_dil, size_t pitch,
                                size_t width, size_t height, float min,
                                float max)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    float* line_harris_im = line(harris_im, y, pitch);
    float* line_harris_dil = line(harris_dil, y, pitch);

    int is_close = abs(line_harris_im[x] - line_harris_dil[x])
        <= (1.0e-8 + 1.0e-5 * abs(line_harris_dil[x]));

    line_harris_dil[x] =
        line_harris_im[x] > (min + 0.5 * (max - min)) ? line_harris_im[x] : 0;
    line_harris_dil[x] = line_harris_dil[x] * (float)is_close;
}

__global__ void best_points(char* harris_resp, point* points, char* values,
                            size_t pitch, size_t width, size_t height,
                            int* count)
{
    int current_count = 0;

    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    float* line_harris_resp = line(harris_resp, y, pitch);
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

    char* harris_buffers;
    size_t pitch_harris_buffers;

    point* points;

    rc = cudaMallocPitch(&image, &pitch_img, width * sizeof(rgb_png), height);
    if (rc)
        abortError("Fail buffer allocation");

    rc = cudaMallocPitch(&buffer, &pitch_buffer, width * sizeof(float), height);
    if (rc)
        abortError("Fail buffer allocation");

    rc = cudaMallocPitch(&harris_buffers, &pitch_harris_buffers,
                         width * sizeof(float), 12 * height);
    if (rc)
        abortError("Fail buffer allocation");

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

    // [im_x, im_y, im_xx, im_xy, im_yy]
    gauss_derivatives<<<dimGrid, dimBlock>>>(buffer, pitch_buffer, width,
                                             height, harris_buffers);

    if (cudaPeekAtLastError())
        abortError("Computation Error");

    // [W_xx, W_xy, W_yy, W_xy_2, W_tr, W_det]
    harris_img<<<dimGrid, dimBlock>>>(harris_buffers, pitch_buffer, width,
                                      height);

    if (cudaPeekAtLastError())
        abortError("Computation Error");

    char* harris_im = harris_buffers + 10 * height * pitch_harris_buffers;

    thrust::device_vector<float> vec(
        (float*)harris_im, (float*)(harris_im + height * pitch_harris_buffers));

    // TODO: use minmax
    float harris_im_min = *thrust::min_element(vec.begin(), vec.end());
    float harris_im_max = *thrust::max_element(vec.begin(), vec.end());

    // threshold<<<dimGrid, dimBlock>>>(harris_im, pitch_buffer, width, height,
    //                                  harris_im_max * 0.1);

    if (cudaPeekAtLastError())
        abortError("Computation Error");

    char* harris_dil = harris_buffers + 11 * height * pitch_harris_buffers;

    rc = cudaMemcpy2D(harris_dil, width * sizeof(float), harris_im,
                      pitch_harris_buffers, width * sizeof(float), height,
                      cudaMemcpyDeviceToDevice);
    if (rc)
        abortError("Unable to copy buffer back to memory");

    dilate<<<dimGrid, dimBlock>>>(harris_dil, harris_im, pitch_buffer, width,
                                  height);

    if (cudaPeekAtLastError())
        abortError("Computation Error");

    harris_response<<<dimGrid, dimBlock>>>(harris_im, harris_dil, pitch_buffer,
                                           width, height, harris_im_min,
                                           harris_im_max);

    char* harris_resp = harris_dil;

    int* count;
    rc = cudaMalloc(&count, 1 * sizeof(int));
    if (rc)
        abortError("Fail buffer allocation");

    rc = cudaMemset(count, 0, 1 * sizeof(int));
    if (rc)
        abortError("Fail buffer memset");

    best_points<<<dimGrid, dimBlock>>>(harris_resp, points, harris_buffers,
                                       pitch_buffer, width, height, count);

    thrust::sort_by_key(
        thrust::device, (float*)harris_buffers,
        (float*)(harris_buffers + height * width * sizeof(float)), points);

    char* result = harris_dil;

    rc = cudaMemcpy2D(out_buffer, width * sizeof(float), result,
                      pitch_harris_buffers, width * sizeof(float), height,
                      cudaMemcpyDeviceToHost);
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

    rc = cudaFree(buffer);
    if (rc)
        abortError("Unable to free memory");

    rc = cudaFree(harris_buffers);
    if (rc)
        abortError("Unable to free memory");
}
