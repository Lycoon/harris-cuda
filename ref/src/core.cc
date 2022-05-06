#include "include/core.hh"

void expm(float ** matrix, int rows, int cols)
{
    for (int y = 0; y < cols; y++)
        for (int x = 0; x < rows; x++)
            matrix[y][x] = exp(matrix[y][x]);
}

void powm(float ** matrix, int rows, int cols, float power)
{
    for (int y = 0; y < cols; y++)
        for (int x = 0; x < rows; x++)
            matrix[y][x] = pow(matrix[y][x], power);
}

void addm(float ** matrix1, float ** matrix2, int rows, int cols)
{
    for (int y = 0; y < cols; y++)
        for (int x = 0; x < rows; x++)
            matrix1[y][x] += matrix2[y][x];
}

void divm(float ** matrix1, int rows, int cols, float scalar)
{
    for (int y = 0; y < cols; y++)
        for (int x = 0; x < rows; x++)
            matrix1[y][x] /= scalar;
}

void mulm(float ** matrix1, int rows, int cols, float scalar)
{
    for (int y = 0; y < cols; y++)
        for (int x = 0; x < rows; x++)
            matrix1[y][x] *= scalar;
}

float ** gauss_kernel(int size)
{
    float *** grid = mgrid(-size, size + 1);
    float ** Y = grid[0];
    float ** X = grid[1];

    // Instantiating 2D array
    float ** gauss = new float*[size];
    for (int i = 0; i < size; i++)
        gauss[i] = new float[size];

    powm(X, size, size, 2);
    divm(X, size, size, pow(2 * 0.33 * size, 2));

    powm(Y, size, size, 2);
    divm(Y, size, size, pow(2 * 0.33 * size, 2));

    addm(X, Y, size, size);
    mulm(X, size, size, -1);
    expm(X, size, size);

    for (int y = 0; y < size; y++)
        for (int x = 0; x < size; x++)
            gauss[y][x] = X[y][x];

    free_grid(grid, size);
    return gauss;
}

float *** mgrid(int start, int end)
{
    int size = end - start;
    float *** grid = new float**[2];

    grid[0] = new float*[size]; // Y
    grid[1] = new float*[size]; // X
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

void free_grid(float *** grid, int size)
{
    for (int i = 0; i < 2; i++)
    {
        for (int y = 0; y < size; y++)
            delete[] grid[i][y];
        delete[] grid[i];
    }
    delete[] grid;
}

void print_grid(float *** grid, int size)
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
        std::cout << "]\n" << ((i >= 1) ? "" : "\n");
    }
}

void print_matrix(float ** matrix, int size)
{
    for (int y = 0; y < size; y++)
    {
        std::cout << ((y <= 0) ? "" : " ") << "[";
        for (int x = 0; x < size; x++)
            std::cout << matrix[y][x] << ((x >= size - 1) ? "" : ", ");

        std::cout << "]" << ((y >= size - 1) ? "" : "\n");
    }
    std::cout << "\n";
}