#pragma once

#include <cmath>
#include <cstring>
#include <functional>
#include <iostream>
#include <memory>
#include <vector>

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

using Point = std::tuple<size_t, size_t>;

class Matrix
{
public:
    Matrix(size_t width_, size_t height_);
    Matrix(Matrix& matrix);

    ~Matrix()
    {
        delete[] this->mat;
    }

    static Matrix* convolve(Matrix& matrix, Matrix& kernel);
    static Matrix* convolve(Matrix& matrix, Matrix& kernel,
                            float init_acc_value,
                            std::function<float(float, float, float)> f);

    static Tuple<Matrix, Matrix> mgrid(int start, int end);
    static Matrix* gauss_kernel(int size);
    static Tuple<Matrix, Matrix> gauss_derivative_kernels(int size);
    static Tuple<Matrix, Matrix> gauss_derivatives(Matrix* image,
                                                   int kernelSize);
    Matrix* harris();

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

    Matrix* is_close(Matrix& m);
    Matrix* is_close(Matrix& m, float rtol, float atol);

    std::vector<Point> points();
    std::vector<Point> points(float min, float max = __FLT_MAX__);

    void print();

    float* operator[](size_t i);

    float min();
    float max();

    void lambda(std::function<float(float)> f);
    void lambda(std::function<float(size_t, size_t)> f);
    void lambda(std::function<float(Matrix*, size_t, size_t)> f);
    void lambda(std::function<float(float, float)> f, Matrix& m);

public:
    size_t width;
    size_t height;

private:
    float* mat;
};
