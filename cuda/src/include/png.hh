#pragma once

#include <cstdlib>
#include <png.h>

class ImagePNG
{
public:
    ImagePNG();
    ImagePNG(size_t height_, size_t width_);
    ImagePNG(ImagePNG& img);

    ~ImagePNG()
    {
        for (size_t i = 0; i < height; i++)
        {
            free(row_pointers[i]);
        }
        free(row_pointers);
    }

    static ImagePNG* read(char* filename);

    ImagePNG* grayscale();

    void draw_disk(size_t x, size_t y);

    void write(char* filename);

public:
    size_t height;
    size_t width;

    png_bytep* row_pointers = NULL;
};

void read_png(char* file_name);
void write_png(char* file_name);
int** to_grayscale();
