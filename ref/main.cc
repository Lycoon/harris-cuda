#include <iostream>
#include "src/include/core.hh"

int main()
{
    std::cout << "Harris corner detector" << "\n";

    int startGrid = -3;
    int endGrid = 5;
    int size = endGrid - startGrid;

    float *** grid = mgrid(startGrid, endGrid);
    print_grid(grid, size);
    free_grid(grid, size);
}
