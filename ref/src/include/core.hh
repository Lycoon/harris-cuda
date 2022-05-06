#pragma once

#include <math.h>
#include <iostream>

float ** gauss_kernel(int size);
float *** mgrid(int start, int end);

void expm(float ** matrix, int rows, int cols);
void powm(float ** matrix, int rows, int cols, float power);
void addm(float ** matrix1, float ** matrix2, int rows, int cols);
void divm(float ** matrix1, int rows, int cols, float scalar);
void mulm(float ** matrix1, int rows, int cols, float scalar);

void free_grid(float *** grid, int size);
void print_grid(float *** grid, int size);
void print_matrix(float ** matrix, int size);