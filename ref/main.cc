#include <iostream>
#include "src/include/core.hh"

int main()
{
    std::cout << "Harris corner detector" << "\n";

    int startGrid = -1;
    int endGrid = 2;
    int size = endGrid - startGrid;

    float *** grid = mgrid(startGrid, endGrid);
    print_grid(grid, size);
    free_grid(grid, size);
}
