#include <iostream>
#include "src/include/core.hh"

int main()
{
    std::cout << "Harris corner detector" << "\n";

    int startGrid = -1;
    int endGrid = 2;
    int size = endGrid - startGrid;

    float ** gauss = gauss_kernel(size);
    print_gaussian(gauss, size);
}
