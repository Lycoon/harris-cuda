#include "include/matrix.hh"

Matrix::Matrix(size_t height_, size_t width_)
    : width(width_)
    , height(height_)
{
    /// Will need padding to 128 / 64 in CUDA
    this->mat = new float[height * width * sizeof(float)];

    auto reset = [](float e) {
        (void)e;
        return 0;
    };
    lambda(reset);
};

Tuple<Matrix, Matrix> Matrix::mgrid(int start, int end)
{
    int size = end - start;
    Matrix* m1 = new Matrix(size, size);
    Matrix* m2 = new Matrix(size, size);

    for (int i = start, y = 0; i < end; i++, y++)
    {
        for (int j = start, x = 0; j < end; j++, x++)
        {
            (*m1)[y][x] = i; // Y
            (*m2)[y][x] = j; // X
        }
    }

    return { m1, m2 };
}

void Matrix::lambda(std::function<float(float)> f)
{
    for (size_t i = 0; i < this->height; i++)
    {
        for (size_t j = 0; j < this->width; j++)
        {
            (*this)[i][j] = f((*this)[i][j]);
        }
    }
}

void Matrix::lambda(std::function<float(float, float)> f, Matrix& m)
{
    for (size_t i = 0; i < this->height; i++)
    {
        for (size_t j = 0; j < this->width; j++)
        {
            (*this)[i][j] = f(m[i][j], (*this)[i][j]);
        }
    }
}

void Matrix::fill(float scalar)
{
    auto fill_ = [scalar](float e) {
        (void)e;
        return scalar;
    };
    lambda(fill_);
}

void Matrix::exp()
{
    auto exp_ = [](float e) { return std::exp(e); };
    lambda(exp_);
}

void Matrix::pow(float scalar)
{
    auto pow_ = [scalar](float e) { return std::pow(e, scalar); };
    lambda(pow_);
}

void Matrix::div(float scalar)
{
    mul(1. / scalar);
}

void Matrix::mul(float scalar)
{
    auto mul_ = [scalar](float e) { return e * scalar; };
    lambda(mul_);
}

void Matrix::add(float scalar)
{
    auto add_ = [scalar](float e) { return e + scalar; };
    lambda(add_);
}

void Matrix::sub(float scalar)
{
    add(-scalar);
}

void Matrix::pow(Matrix& m)
{
    auto pow_ = [](float e, float m) { return std::pow(m, e); };
    lambda(pow_, m);
}

void Matrix::div(Matrix& m)
{
    auto div_ = [](float e, float m) { return m / e; };
    lambda(div_, m);
}

void Matrix::mul(Matrix& m)
{
    auto mul_ = [](float e, float m) { return m * e; };
    lambda(mul_, m);
}

void Matrix::add(Matrix& m)
{
    auto add_ = [](float e, float m) { return m + e; };
    lambda(add_, m);
}

void Matrix::sub(Matrix& m)
{
    auto sub_ = [](float e, float m) { return m - e; };
    lambda(sub_, m);
}

Matrix* Matrix::convolve(int** image, Matrix& kernel, int imgWidth,
                         int imgHeight, int kSize)
{
    // Instantiating convoluted image
    Matrix* res = new Matrix(imgHeight, imgWidth);

    for (int imgY = 0; imgY < imgHeight; imgY++)
    {
        for (int imgX = 0; imgX < imgWidth; imgX++)
        {
            float acc = 0;
            for (int kY = -kSize / 2, kI = 0; kY < kSize / 2; kY++, kI++)
            {
                for (int kX = -kSize / 2, kJ = 0; kX < kSize / 2; kX++, kJ++)
                {
                    if (imgY + kY >= 0 && imgY + kY < imgHeight
                        && imgX + kX >= 0 && imgX + kX < imgWidth)
                        acc += image[imgY + kY][imgX + kX] * kernel[kJ][kI];
                }
            }
            (*res)[imgY][imgX] = acc;
        }
    }

    return res;
}

Tuple<Matrix, Matrix> Matrix::gauss_derivatives(int** image, int imgWidth,
                                                int imgHeight, int size)
{
    auto gauss = Matrix::gauss_derivative_kernels(size);

    auto imx = Matrix::convolve(image, *gauss.first, imgWidth, imgHeight, size);
    auto imy =
        Matrix::convolve(image, *gauss.second, imgWidth, imgHeight, size);

    return { imx, imy };
}

Tuple<Matrix, Matrix> Matrix::gauss_derivative_kernels(int size)
{
    Tuple<Matrix, Matrix> grid = Matrix::mgrid(-size, size + 1);
    Matrix* Y = grid.first;
    Matrix* X = grid.second;

    Matrix* gx = Matrix::gauss_kernel(size);
    Matrix* gy = Matrix::gauss_kernel(size);

    Y->mul(-1);
    X->mul(-1);
    gy->mul(*Y);
    gx->mul(*X);

    return { gx, gy };
}

Matrix* Matrix::gauss_kernel(int size)
{
    int mSize = size * 2 + 1; // matrix size
    Tuple<Matrix, Matrix> grid = Matrix::mgrid(-size, size + 1);

    Matrix* Y = grid.first;
    Matrix* X = grid.second;

    X->pow(2);
    X->div(2 * powf(0.33 * size, 2));

    Y->pow(2);
    Y->div(2 * powf(0.33 * size, 2));

    X->add(*Y);
    X->mul(-1);
    X->exp();

    Matrix* gauss = new Matrix(mSize, mSize);

    for (int y = 0; y < mSize; y++)
        for (int x = 0; x < mSize; x++)
            (*gauss)[y][x] = (*X)[y][x];

    return gauss;
}

void Matrix::print()
{
    for (size_t i = 0; i < this->height; i++)
    {
        for (size_t j = 0; j < this->width - 1; j++)
        {
            std::cout << (*this)[i][j] << " ";
        }
        std::cout << (*this)[i][this->width - 1] << "\n";
    }
}

float* Matrix::operator[](size_t i)
{
    return this->mat + (i * this->width);
}
