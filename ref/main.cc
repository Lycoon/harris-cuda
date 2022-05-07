#include <iostream>
#include "src/include/core.hh"

int main()
{
    std::cout << "Harris corner detector"
              << "\n";

    float **gauss = gauss_kernel(1);
    print_matrix(gauss, 3, 3);
}
