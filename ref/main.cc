#include <iostream>
#include "src/include/core.hh"

int main()
{
    std::cout << "Harris corner detector"
              << "\n";

    float **gauss = gauss_kernel(1);
    // print_matrix(gauss, 3, 3);

    float ***derivative = gauss_derivative_kernels(1);
    print_matrix(derivative[1], 3, 3);
    free_matrix3(derivative, 3);
}
