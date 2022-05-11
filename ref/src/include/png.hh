#pragma once

#include <png.h>

#include "matrix.hh"

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
    static void write_matrix(char* filename, Matrix* image);
    static void write_matrix(char* filename, Matrix* image, Matrix* points);
    static void write_matrix(char* filename, Matrix* image, Matrix* points1,
                             Matrix* points2);

    ImagePNG* grayscale();

    Matrix* grayscale_matrix();
    void write(char* filename);

public:
    size_t height;
    size_t width;

private:
    png_bytep* row_pointers = NULL;
};

void read_png(char* file_name);
void write_png(char* file_name);
void save(Matrix* image, int width, int height);
int** to_grayscale();
