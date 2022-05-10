#include <iostream>

#include "src/include/matrix.hh"
#include "src/include/morphology.hh"
#include "src/include/png.hh"

int main()
{
    std::cout << "Harris corner detector"
              << "\n";

    const auto DERIVATIVE_KERNEL_SIZE = 3;
    const auto OPENING_SIZE = 3;

    auto image = ImagePNG::read((char*)"../../twin_it/bubbles_200dpi/b003.png");
    auto gray = image->grayscale_matrix();
    ImagePNG::write_matrix("gray.png", gray);

    auto derivatives = Matrix::gauss_derivatives(gray, DERIVATIVE_KERNEL_SIZE);

    ImagePNG::write_matrix("gauss_x.png", derivatives.first);
    ImagePNG::write_matrix("gauss_y.png", derivatives.second);

    auto harris = gray->harris();
    ImagePNG::write_matrix("harris.png", harris);

    auto mask = new Matrix(*gray);
    mask->lambda([](float e) { return e > 0.00001 ? 1. : 0.; });
    auto mask_eroded = eroded_mask(*gray);

    ImagePNG::write_matrix("mask.png", mask);
    ImagePNG::write_matrix("mask_eroded.png", mask_eroded);

    delete image;
    delete gray;
    delete harris;
    delete mask;
    delete mask_eroded;
}
