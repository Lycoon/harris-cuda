#include <cstddef>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>

#include "include/harris.hh"
#include "include/png.hh"

void write_png(png_bytepp buffer, int width, int height, const char* filename)
{
    FILE* fp = fopen(filename, "wb");
    png_structp png_ptr =
        png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    png_init_io(png_ptr, fp);

    png_infop info_ptr = png_create_info_struct(png_ptr);
    png_set_IHDR(png_ptr, info_ptr, width, height, 8, PNG_COLOR_TYPE_RGB,
                 PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_BASE,
                 PNG_FILTER_TYPE_BASE);

    png_write_info(png_ptr, info_ptr);

    png_set_rows(png_ptr, info_ptr, buffer);
    png_write_png(png_ptr, info_ptr, PNG_TRANSFORM_IDENTITY, NULL);
    png_destroy_write_struct(&png_ptr, &info_ptr);

    fclose(fp);
}

int main(int argc, char** argv)
{
    if (argc != 2)
    {
        std::cerr << "Usage: ./harris <image>" << std::endl;
        exit(3);
    }

    auto image = ImagePNG::read(argv[1]);

    // Create buffer (will need pitch)
    int stride = image->width * sizeof(rgb_png);
    auto buffer = std::make_unique<std::byte[]>(image->height * stride);
    auto buffer_out = std::make_unique<std::byte[]>(image->height * image->width
                                                    * sizeof(float));
    auto point_out = std::make_unique<std::byte[]>(2000 * sizeof(point));

    for (size_t i = 0; i < image->height; i++)
    {
        memcpy(buffer.get() + i * stride, image->row_pointers[i], stride);
    }

    size_t nb_points = 0;
    // harris
    harris(reinterpret_cast<char*>(buffer.get()),
           reinterpret_cast<char*>(buffer_out.get()),
           reinterpret_cast<point*>(point_out.get()), &nb_points, image->width,
           image->height, stride);

    std::stringstream output;
    for (size_t i = 0; i < nb_points; i++)
    {
        auto p = ((point*)point_out.get())[i];
        image->draw_disk(p.x, p.y);
        output << "x: " << p.x << " | y: " << p.y << "\n";
    }

    image->write("output.png");

    std::cout << output.str();

    std::ofstream output_file("output.txt");
    output_file << output.str();
    output_file.close();

    delete image;
}
