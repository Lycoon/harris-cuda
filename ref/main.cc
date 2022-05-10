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
    auto mask_eroded = eroded_mask(*gray, 10);

    ImagePNG::write_matrix("mask.png", mask);
    ImagePNG::write_matrix("mask_eroded.png", mask_eroded);

    const auto THRESHOLD = 0.1;
    auto corner_threshold = harris->max() * THRESHOLD;

    auto harris_mask = new Matrix(*harris);
    harris_mask->lambda(
        [corner_threshold](float e) { return e > corner_threshold ? 1. : 0.; });

    auto harris_mask_eroded = new Matrix(*harris_mask);
    harris_mask_eroded->mul(*mask_eroded);

    ImagePNG::write_matrix("harris_mask.png", harris_mask);
    ImagePNG::write_matrix("harris_mask_eroded.png", harris_mask_eroded);

    auto ell_kernel = ellipse_kernel(20, 20);
    auto harris_dilated = dilation(*harris, *ell_kernel);

    ImagePNG::write_matrix("harris_dilated.png", harris_dilated);

    delete image;
    delete gray;
    delete harris;
    delete mask;
    delete mask_eroded;
    delete harris_mask;
    delete harris_mask_eroded;
    delete ell_kernel;
    delete harris_dilated;
}
