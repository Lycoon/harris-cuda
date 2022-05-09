#include <iostream>

#include "src/include/core.hh"
#include "src/include/png.hh"

int main()
{
    std::cout << "Harris corner detector"
              << "\n";

    // float **gauss = gauss_kernel(1);
    // print_matrix(gauss, 3, 3);

    float ***derivative = gauss_derivative_kernels(1);
    print_matrix(derivative[1], 3, 3);
    free_matrix3(derivative, 3);

    read_png((char *)"../../twin_it/bubbles_200dpi/b003.png");
    int **grayscaled = to_grayscale();

    float ***derivatives = gauss_derivatives(grayscaled, 254, 251, 3);
    print_matrix(derivatives[0], 251, 254);
    save(derivatives[0], 254, 251);
}
