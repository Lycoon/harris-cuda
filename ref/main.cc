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
    auto detection = harris_points(*harris);

    auto points = detection->points(0.5, 1.5);
    std::cout << points.size() << std::endl;

    for (auto it : points)
    {
        std::cout << std::get<0>(it) << " " << std::get<1>(it) << std::endl;
    }

    ImagePNG::write_matrix("gray.png", gray);
    ImagePNG::write_matrix("detect.png", detection);

    delete image;
    delete gray;
    delete harris;
    delete detection;
}
