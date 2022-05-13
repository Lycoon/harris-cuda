#include <cstddef>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>

#include "include/harris.hh"
#include "include/png.hh"

void write_png(png_bytepp buffer, int width, int height, int stride,
               const char* filename)
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

// Usage: ./mandel
int main(int argc, char** argv)
{
    (void)argc;
    (void)argv;

    std::string filename = "output.png";

    auto image = ImagePNG::read((char*)"../twin_it/bubbles_200dpi/b006.png");

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

    int nb_points = 0;
    // harris
    harris(reinterpret_cast<char*>(buffer.get()),
           reinterpret_cast<char*>(buffer_out.get()),
           reinterpret_cast<point*>(point_out.get()), &nb_points, image->width,
           image->height, stride);

    float min = std::numeric_limits<float>::max();
    float max = std::numeric_limits<float>::min();
    for (size_t y = 0; y < image->height; y++)
    {
        for (size_t x = 0; x < image->width; x++)
        {
            float tmp = ((float*)buffer_out.get())[y * image->width + x];
            min = std::min(min, tmp);
            max = std::max(max, tmp);
        }
    }

    // Write image data
    for (size_t y = 0; y < image->height; y++)
    {
        for (size_t x = 0; x < image->width; x++)
        {
            float grey =
                ((((float*)buffer_out.get())[y * image->width + x] - min)
                 * 255.f)
                / (max - min);
            for (size_t k = 0; k < 3; k++)
                image->row_pointers[y][x * 3 + k] = static_cast<png_byte>(grey);
        }
    }

    write_png(image->row_pointers, image->width, image->height, stride,
              filename.c_str());

    std::cout << "nb_points: " << nb_points << "\n";
    for (size_t i = 0; i < nb_points; i++)
    {
        std::cout << "x: " << ((point*)point_out.get())[i].x
                  << " | y: " << ((point*)point_out.get())[i].y << "\n";
    }

    delete image;
}
