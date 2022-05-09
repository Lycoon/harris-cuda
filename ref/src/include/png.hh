#pragma once

#include <png.h>

void read_png(char *file_name);
void write_png(char *file_name);
void save(float **image, int width, int height);
int **to_grayscale();
