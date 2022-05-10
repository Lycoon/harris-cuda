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

    auto image = ImagePNG::read((char*)"../twin_it/bubbles_200dpi/b003.png");
    auto gray = image->grayscale_matrix();

    auto derivatives = Matrix::gauss_derivatives(gray, 3);

    ImagePNG::write_matrix("gray.png", gray);
    ImagePNG::write_matrix("gauss.png", derivatives.first);

    delete gray;
    delete image;

    // auto gauss = Matrix::gauss_derivative_kernels(1);
    // gauss.first->print();
    // std::cout << "\n";
    // gauss.second->print();
}
