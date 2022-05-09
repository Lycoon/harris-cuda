#include "include/png.hh"

#include <algorithm>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// png_byte color_type;
// png_byte bit_depth;
png_infop info_ptr;
png_bytep* row_pointers = NULL;
png_structp png_ptr;

void read_png(char* file_name)
{
    FILE* fp = fopen(file_name, "rb");
    png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    info_ptr = png_create_info_struct(png_ptr);
    png_init_io(png_ptr, fp);
    png_read_png(png_ptr, info_ptr, PNG_TRANSFORM_IDENTITY, NULL);
    row_pointers = png_get_rows(png_ptr, info_ptr);
    // png_destroy_read_struct(&png_ptr, NULL, NULL);
    fclose(fp);
}

void write_png(char* file_name)
{
    FILE* fp = fopen(file_name, "wb");
    png_structp png_ptr =
        png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    png_init_io(png_ptr, fp);
    png_set_rows(png_ptr, info_ptr, row_pointers);
    png_write_png(png_ptr, info_ptr, PNG_TRANSFORM_IDENTITY, NULL);
    png_destroy_write_struct(&png_ptr, &info_ptr);
    fclose(fp);
}

void save(Matrix* image, int width, int height)
{
    float max = 0, min = 0;
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            max = std::max((*image)[y][x], max);
            min = std::min((*image)[y][x], min);
        }
    }

    for (int y = 0; y < height; y++)
    {
        png_bytep row = row_pointers[y];
        memset(row, 0, width);
        for (int x = 0; x < width; x++)
        {
            png_bytep px = &(row[x * 3]);
            auto val = (((*image)[y][x] - min) * 255.) / (max - min);

            px[0] = val;
            px[1] = val;
            px[2] = val;
        }
    }

    write_png("bruh.png");
}

int** to_grayscale()
{
    auto width = png_get_image_width(png_ptr, info_ptr);
    auto height = png_get_image_height(png_ptr, info_ptr);

    int** res = new int*[height];
    for (int y = 0; y < height; y++)
        res[y] = new int[width];

    for (int y = 0; y < height; y++)
    {
        png_bytep row = row_pointers[y];
        for (int x = 0; x < width; x++)
        {
            png_bytep px = &(row[x * 3]);
            res[y][x] = px[0] * 0.299 + px[1] * 0.587 + px[2] * 0.114;
        }
    }

    return res;
}
