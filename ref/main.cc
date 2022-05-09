#include <iostream>

#include "src/include/matrix.hh"
#include "src/include/png.hh"

int main()
{
    std::cout << "Harris corner detector"
              << "\n";

    // float **gauss = gauss_kernel(1);
    // print_matrix(gauss, 3, 3);

    auto derivative = Matrix::gauss_derivative_kernels(1);
    derivative.second->print();

    read_png((char*)"../twin_it/bubbles_200dpi/b003.png");
    int** grayscaled = to_grayscale();

    auto derivatives = Matrix::gauss_derivatives(grayscaled, 254, 251, 3);
    // derivatives.first->print();
    save(derivatives.first, 254, 251);

    for (int y = 0; y < 251; y++)
        delete[] grayscaled[y];
    delete[] grayscaled;

    // auto gauss = Matrix::gauss_derivative_kernels(1);
    // gauss.first->print();
    // std::cout << "\n";
    // gauss.second->print();
}
