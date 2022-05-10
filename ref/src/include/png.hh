#pragma once

#include <png.h>

#include "matrix.hh"

class ImagePNG
{
public:
    ImagePNG();
    ImagePNG(size_t height_, size_t width_);
    // ImagePNG(ImagePNG& cpy);

    ~ImagePNG()
    {
        free(info_ptr);
        for (size_t i = 0; i < height; i++)
        {
            free(row_pointers[i]);
        }
        free(row_pointers);
    }

    static ImagePNG* read(char* filename);
    ImagePNG* grayscale();
    Matrix* grayscale_matrix();
    void write(char* filename);
    static void write_matrix(char* filename, Matrix* image);

    void set_info(png_infop info_ptr)
    {
        this->info_ptr = info_ptr;
    }

    void set_rows_ptr(png_bytep* row_pointers)
    {
        this->row_pointers = row_pointers;
    }

public:
    size_t height;
    size_t width;

private:
    png_infop info_ptr;
    png_bytep* row_pointers = NULL;
};

void read_png(char* file_name);
void write_png(char* file_name);
void save(Matrix* image, int width, int height);
int** to_grayscale();
