#pragma once

#include <png.h>

#include "matrix.hh"

void read_png(char* file_name);
void write_png(char* file_name);
void save(Matrix* image, int width, int height);
int** to_grayscale();
