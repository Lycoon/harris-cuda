#include <fstream>
#include <iostream>
#include <sstream>

#include "src/include/matrix.hh"
#include "src/include/morphology.hh"
#include "src/include/png.hh"

int main(int argc, char** argv)
{
    if (argc != 2)
    {
        std::cerr << "Usage: ./harris <image>" << std::endl;
        exit(3);
    }

    auto image = ImagePNG::read(argv[1]);

    auto gray = image->grayscale_matrix();
    auto harris = gray->harris();

    auto harris_resp = harris_response(*harris);
    auto best = best_harris_points(*harris);

    std::stringstream output;
    for (auto p : best)
    {
        auto y = std::get<0>(p);
        auto x = std::get<1>(p);

        image->draw_disk(x, y);
        output << "x: " << x << " | y: " << y << "\n";
    }

    image->write("output.png");

    std::cout << output.str();

    std::ofstream output_file("output.txt");
    output_file << output.str();
    output_file.close();

    delete image;
    delete gray;
    delete harris;
    delete harris_resp;
}
