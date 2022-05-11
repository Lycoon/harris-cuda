#include <cstddef>
#include <iostream>
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

    // Create buffer
    int stride = image->width * sizeof(png_byte) * 3;

    std::cout << "ok maguele" << std::endl;
    // harris
    harris(reinterpret_cast<char*>(image->row_pointers), image->width,
           image->height, stride);

    // Save
    write_png(image->row_pointers, image->width, image->height, stride,
              filename.c_str());

    delete image;
}
