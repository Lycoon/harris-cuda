#include "include/core.hh"

float ** gauss_kernel(int size)
{
    float *** grid = mgrid(-size, size + 1);
    float ** Y = grid[0];
    float ** X = grid[1];

    // Instantiating 2D array
    float ** gauss = new float*[size];
    for (int i = 0; i < size; i++)
        gauss[i] = new float[size];

    for (int y = 0; y < size; y++)
    {
        for (int x = 0; x < size; x++)
        {
            float val = exp(-(pow(X[y][x], 2) / (2*pow(0.33 * size, 2)) + pow(Y[y][x], 2) / (2*pow(0.33 * size, 2))));
            gauss[y][x] = val;
        }
    }

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

    int y = 0, x = 0;
    for (int i = start; i < end; i++)
    {
        for (int j = start; j < end; j++, x++)
            grid[0][y][x] = i; // Y

        for (int j = start, x = 0; j < end; j++, x++)
            grid[1][y][x] = j; // X

        y++;
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
            {
                std::cout << grid[i][y][x] << ((x >= size - 1) ? "" : ", ");
            }
            std::cout << "]" << ((y >= size - 1) ? "" : "\n");
        }
        std::cout << "]\n" << ((i >= 1) ? "" : "\n");
    }
}