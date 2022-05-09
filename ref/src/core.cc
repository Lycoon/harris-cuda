#include "include/core.hh"

void expm(float **matrix, int size)
{
    expm(matrix, size, size);
}

void powm(float **matrix, int size, float power)
{
    powm(matrix, size, size, power);
}

void mulm(float **matrix, int size, float scalar)
{
    mulm(matrix, size, size, scalar);
}

void divm(float **matrix, int size, float scalar)
{
    divm(matrix, size, size, scalar);
}

void expm(float **matrix, int rows, int cols)
{
    for (int y = 0; y < rows; y++)
        for (int x = 0; x < cols; x++)
            matrix[y][x] = exp(matrix[y][x]);
}

void powm(float **matrix, int rows, int cols, float power)
{
    for (int y = 0; y < rows; y++)
        for (int x = 0; x < cols; x++)
            matrix[y][x] = powf(matrix[y][x], power);
}

void mulm(float **matrix1, int rows, int cols, float scalar)
{
    for (int y = 0; y < rows; y++)
        for (int x = 0; x < cols; x++)
            matrix1[y][x] *= scalar;
}

void mulm_element_wise(float **matrix1, float **matrix2, int size)
{
    for (int y = 0; y < size; y++)
        for (int x = 0; x < size; x++)
            matrix1[y][x] *= matrix2[y][x];
}

void divm(float **matrix1, int rows, int cols, float scalar)
{
    for (int y = 0; y < rows; y++)
        for (int x = 0; x < cols; x++)
            matrix1[y][x] /= scalar;
}

void addm(float **matrix1, float **matrix2, int size)
{
    for (int y = 0; y < size; y++)
        for (int x = 0; x < size; x++)
            matrix1[y][x] += matrix2[y][x];
}

int **convolve(int **image, float **kernel, int imgWidth, int imgHeight, int kSize)
{
    // Instantiating convoluted image
    int **res = new int *[imgHeight];
    for (int y = 0; y < imgHeight; y++)
        res[y] = new int[imgWidth];

    for (int imgY = 0; imgY < imgHeight; imgY++)
    {
        for (int imgX = 0; imgX < imgWidth; imgX++)
        {
            float acc = 0;
            for (int kY = -kSize / 2, kI = 0; kY < kSize / 2; kY++, kI++)
            {
                for (int kX = -kSize / 2, kJ = 0; kX < kSize / 2; kX++, kJ++)
                {
                    if (imgY + kY >= 0 && imgY + kY < imgHeight && imgX + kX >= 0 && imgX + kX < imgWidth)
                        acc += image[imgY + kY][imgX + kX] * kernel[kI][kJ];
                }
            }
            res[imgY][imgX] = acc;
        }
    }

    return res;
}

int ***gauss_derivatives(int **image, int imgWidth, int imgHeight, int size)
{
    auto g_xy = gauss_derivative_kernels(size);

    auto imx = convolve(image, g_xy[0], imgWidth, imgHeight, size);
    auto imy = convolve(image, g_xy[1], imgWidth, imgHeight, size);

    int ***res = new int **[2];
    res[0] = imx;
    res[1] = imy;

    return res;
}

float ***gauss_derivative_kernels(int size)
{
    int mSize = size * 2 + 1; // matrix size
    float ***grid = mgrid(-size, size + 1);
    float **Y = grid[0];
    float **X = grid[1];

    float **gx = gauss_kernel(size);
    float **gy = gauss_kernel(size);

    mulm(Y, mSize, -1);
    mulm(X, mSize, -1);
    mulm_element_wise(gy, Y, mSize);
    mulm_element_wise(gx, X, mSize);

    float ***res = new float **[2];
    res[0] = gx;
    res[1] = gy;

    free_matrix3(grid, mSize);
    return res;
}

float **gauss_kernel(int size)
{
    int mSize = size * 2 + 1; // matrix size
    float ***grid = mgrid(-size, size + 1);
    float **Y = grid[0];
    float **X = grid[1];

    // Instantiating 2D array
    float **gauss = new float *[mSize];
    for (int i = 0; i < mSize; i++)
        gauss[i] = new float[mSize];

    powm(X, mSize, 2);
    divm(X, mSize, 2 * powf(0.33 * size, 2));

    powm(Y, mSize, 2);
    divm(Y, mSize, 2 * powf(0.33 * size, 2));

    addm(X, Y, mSize);
    mulm(X, mSize, -1);
    expm(X, mSize);

    for (int y = 0; y < mSize; y++)
        for (int x = 0; x < mSize; x++)
            gauss[y][x] = X[y][x];

    free_matrix3(grid, mSize);
    return gauss;
}

float ***mgrid(int start, int end)
{
    int size = end - start;
    float ***grid = new float **[2];

    grid[0] = new float *[size]; // Y
    grid[1] = new float *[size]; // X
    for (int i = 0; i < size; i++)
    {
        grid[0][i] = new float[size];
        grid[1][i] = new float[size];
    }

    for (int i = start, y = 0; i < end; i++, y++)
    {
        for (int j = start, x = 0; j < end; j++, x++)
        {
            grid[0][y][x] = i; // Y
            grid[1][y][x] = j; // X
        }
    }

    return grid;
}

void free_matrix3(float ***grid, int size)
{
    for (int i = 0; i < 2; i++)
    {
        for (int y = 0; y < size; y++)
            delete[] grid[i][y];
        delete[] grid[i];
    }
    delete[] grid;
}

void print_grid(float ***grid, int size)
{
    for (int i = 0; i < 2; i++)
    {
        std::cout << "[";
        for (int y = 0; y < size; y++)
        {
            std::cout << ((y <= 0) ? "" : " ") << "[";
            for (int x = 0; x < size; x++)
                std::cout << grid[i][y][x] << ((x >= size - 1) ? "" : ", ");

            std::cout << "]" << ((y >= size - 1) ? "" : "\n");
        }
        std::cout << "]\n"
                  << ((i >= 1) ? "" : "\n");
    }
}

void print_matrix(float **matrix, int rows, int cols)
{
    for (int y = 0; y < rows; y++)
    {
        std::cout << "[";
        for (int x = 0; x < cols; x++)
            std::cout << matrix[y][x] << ((x >= cols - 1) ? "" : ", ");

        std::cout << "]" << ((y >= rows - 1) ? "" : "\n");
    }
    std::cout << "\n";
}