#include <iostream>

#include "src/include/matrix.hh"
#include "src/include/png.hh"

int main()
{
    std::cout << "Harris corner detector"
              << "\n";

    auto derivative = Matrix::gauss_derivative_kernels(1);
    std::cout << "gauss x kernel: " << std::endl;
    derivative.first->print();
    std::cout << "gauss y kernel: " << std::endl;
    derivative.second->print();

    auto image = ImagePNG::read((char*)"../../twin_it/bubbles_200dpi/b003.png");
    auto gray = image->grayscale_matrix();

    auto derivatives = Matrix::gauss_derivatives(gray, 3);

    ImagePNG::write_matrix("gray.png", gray);
    ImagePNG::write_matrix("gauss_x.png", derivatives.first);
    ImagePNG::write_matrix("gauss_y.png", derivatives.second);

    delete gray;
    delete image;
}
