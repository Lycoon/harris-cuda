#pragma once

#include <math.h>

float ** gauss_kernel(int size);
float *** mgrid(int start, int end);
void free_grid(float *** grid, int size);