#include "include/png.hh"

#include <algorithm>
#include <iostream>
#include <limits>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

ImagePNG::ImagePNG()
    : height(0)
    , width(0){};

ImagePNG::ImagePNG(size_t height_, size_t width_)
    : height(height_)
    , width(width_){};

ImagePNG::ImagePNG(ImagePNG& img)
{
    this->height = img.height;
    this->width = img.width;

    auto row_pointers = (png_bytepp)malloc(img.height * sizeof(png_bytep));
    for (size_t i = 0; i < img.height; i++)
    {
        row_pointers[i] = (png_bytep)malloc(3 * img.width * sizeof(png_byte));
        memcpy(row_pointers[i], img.row_pointers[i],
               3 * img.width * sizeof(png_byte));
    }

    this->row_pointers = row_pointers;
}

ImagePNG* ImagePNG::read(char* filename)
{
    ImagePNG* image = new ImagePNG();

    FILE* fp = fopen(filename, "rb");
    if (!fp)
    {
        std::cerr << filename << " not found" << std::endl;
        exit(3);
    }

    auto png_ptr =
        png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    auto info_ptr = png_create_info_struct(png_ptr);
    png_init_io(png_ptr, fp);
    png_read_png(png_ptr, info_ptr, PNG_TRANSFORM_IDENTITY, NULL);

    auto row_pointers = png_get_rows(png_ptr, info_ptr);

    image->width = png_get_image_width(png_ptr, info_ptr);
    image->height = png_get_image_height(png_ptr, info_ptr);

    image->row_pointers = row_pointers;

    free(info_ptr);
    png_destroy_read_struct(&png_ptr, NULL, NULL);
    fclose(fp);

    return image;
}

void ImagePNG::write(char filename[])
{
    FILE* fp = fopen(filename, "wb");
    png_structp png_ptr =
        png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    png_init_io(png_ptr, fp);

    png_infop info_ptr = png_create_info_struct(png_ptr);
    png_set_IHDR(png_ptr, info_ptr, this->width, this->height, 8,
                 PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_BASE, PNG_FILTER_TYPE_BASE);

    png_write_info(png_ptr, info_ptr);

    png_set_rows(png_ptr, info_ptr, this->row_pointers);
    png_write_png(png_ptr, info_ptr, PNG_TRANSFORM_IDENTITY, NULL);
    png_destroy_write_struct(&png_ptr, &info_ptr);

    fclose(fp);
}

ImagePNG* ImagePNG::grayscale()
{
    ImagePNG* image = new ImagePNG(*this);

    for (size_t y = 0; y < this->height; y++)
    {
        png_bytep row = this->row_pointers[y];
        auto new_row = image->row_pointers[y];

        for (size_t x = 0; x < this->width; x++)
        {
            png_bytep px = &(row[x * 3]);
            png_bytep new_px = &(new_row[x * 3]);

            auto gray = px[0] * 0.299 + px[1] * 0.587 + px[2] * 0.114;
            new_px[0] = new_px[1] = new_px[2] = gray;
        }
    }

    return image;
}

const uint16_t DISK_POINTS[] = {
    (0 << 8) | 1, (0 << 8) | 2, (0 << 8) | 3, (1 << 8) | 0, (1 << 8) | 1,
    (1 << 8) | 2, (1 << 8) | 3, (1 << 8) | 4, (2 << 8) | 0, (2 << 8) | 1,
    (2 << 8) | 2, (2 << 8) | 3, (2 << 8) | 4, (3 << 8) | 0, (3 << 8) | 1,
    (3 << 8) | 2, (3 << 8) | 3, (3 << 8) | 4, (4 << 8) | 1, (4 << 8) | 2,
    (4 << 8) | 3,
};

void ImagePNG::draw_disk(size_t x, size_t y)
{
    int offSet = 2;
    for (uint16_t p : DISK_POINTS)
    {
        int kX = (int)(p & 0xFF) - offSet + x;
        int kY = (int)((p >> 8) & 0xFF) - offSet + y;

        // FIXME: "free(): invalid next size (normal)"
        this->row_pointers[kY][kX * 3 + 0] = static_cast<png_byte>(255);
        this->row_pointers[kY][kX * 3 + 1] = static_cast<png_byte>(0);
        this->row_pointers[kY][kX * 3 + 2] = static_cast<png_byte>(0);
    }
}
