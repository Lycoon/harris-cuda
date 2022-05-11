#include <iostream>

#include "src/include/matrix.hh"
#include "src/include/morphology.hh"
#include "src/include/png.hh"

int main()
{
    std::cout << "Harris corner detector"
              << "\n";

    auto image = ImagePNG::read((char*)"../twin_it/bubbles_200dpi/b006.png");

    auto gray = image->grayscale_matrix();
    auto harris = gray->harris();

    auto harris_resp = harris_response(*harris);
    auto best = best_harris_points(*harris);

    ImagePNG::write_matrix("gray.png", gray);
    ImagePNG::write_matrix("harris.png", harris_resp);

    delete image;
    delete gray;
    delete harris;
    delete harris_resp;
}
