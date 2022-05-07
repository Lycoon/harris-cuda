#pragma once

#include <math.h>
#include <iostream>

float **gauss_kernel(int size);
float ***mgrid(int start, int end);

// General matrix operations
void expm(float **matrix, int rows, int cols);
void powm(float **matrix, int rows, int cols, float power);
void divm(float **matrix1, int rows, int cols, float scalar);
void mulm(float **matrix1, int rows, int cols, float scalar);

// Square-matrix operations
void expm(float **matrix, int size);
void powm(float **matrix, int size, float power);
void mulm(float **matrix, int size, float scalar);
void divm(float **matrix, int size, float scalar);
void addm(float **matrix1, float **matrix2, int size);

void free_grid(float ***grid, int size);
void print_grid(float ***grid, int size);
void print_matrix(float **matrix, int rows, int cols);