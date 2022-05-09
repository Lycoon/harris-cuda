#pragma once

#include <cmath>
#include <functional>
#include <iostream>
#include <memory>

template <typename F, typename S>
class Tuple
{
public:
    Tuple(F* f, S* s)
        : first(f)
        , second(s)
    {}

    ~Tuple()
    {
        delete first;
        delete second;
    };

    F* first;
    S* second;
};

class Matrix
{
public:
    Matrix(size_t width_, size_t height_);
    ~Matrix()
    {
        delete[] this->mat;
    }

    static Tuple<Matrix, Matrix> mgrid(int start, int end);
    static Matrix* gauss_kernel(int size);
    static Tuple<Matrix, Matrix> gauss_derivative_kernels(int size);
    // static Tuple<Matrix, Matrix> gauss_derivatives(int** image, int imgWidth,
    //                                                int imgHeight, int size);
    static Matrix* convolve(int** image, Matrix& kernel, int imgWidth,
                            int imgHeight, int kSize);

    void exp();
    void pow(float power);
    void div(float scalar);
    void mul(float scalar);
    void add(float scalar);
    void sub(float scalar);

    void fill(float scalar);

    void pow(Matrix& m);
    void div(Matrix& m);
    void mul(Matrix& m);
    void add(Matrix& m);
    void sub(Matrix& m);

    void print();

    float* operator[](size_t i);

private:
    void lambda(std::function<float(float)> f);
    void lambda(std::function<float(float, float)> f, Matrix& m);

    float* mat;
    size_t width;
    size_t height;
};
