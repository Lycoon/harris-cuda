#include <iostream>

#include "src/include/matrix.hh"
#include "src/include/png.hh"

int main()
{
    std::cout << "Harris corner detector"
              << "\n";

    const auto DERIVATIVE_KERNEL_SIZE = 3;
    const auto OPENING_SIZE = 3;

    auto image = ImagePNG::read((char*)"../twin_it/bubbles_200dpi/b003.png");
    auto gray = image->grayscale_matrix();
    ImagePNG::write_matrix("gray.png", gray);

    auto derivatives = Matrix::gauss_derivatives(gray, DERIVATIVE_KERNEL_SIZE);

    ImagePNG::write_matrix("gauss_x.png", derivatives.first);
    ImagePNG::write_matrix("gauss_y.png", derivatives.second);

    auto harris = gray->harris();
    ImagePNG::write_matrix("harris.png", harris);

    delete image;
    delete gray;
    delete harris;
}
