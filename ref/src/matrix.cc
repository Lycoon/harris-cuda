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

Matrix::Matrix(Matrix& matrix)
    : width(matrix.width)
    , height(matrix.height)
{
    this->mat = new float[height * width * sizeof(float)];
    std::memcpy(this->mat, matrix.mat, height * width * sizeof(float));
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

void Matrix::lambda(std::function<float(size_t, size_t)> f)
{
    for (size_t i = 0; i < this->height; i++)
    {
        for (size_t j = 0; j < this->width; j++)
        {
            (*this)[i][j] = f(i, j);
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

Matrix* Matrix::harris()
{
    const auto DERIVATIVE_KERNEL_SIZE = 3;
    const auto OPENING_SIZE = 3;

    auto derivatives = Matrix::gauss_derivatives(this, DERIVATIVE_KERNEL_SIZE);

    auto gauss = Matrix::gauss_kernel(OPENING_SIZE);

    auto im_xx = new Matrix(*derivatives.first);
    im_xx->mul(*(derivatives.first));
    auto im_xy = new Matrix(*derivatives.first);
    im_xy->mul(*(derivatives.second));
    auto im_yy = new Matrix(*derivatives.second);
    im_yy->mul(*(derivatives.second));

    auto W_xx = Matrix::convolve(*im_xx, *gauss);
    auto W_xy = Matrix::convolve(*im_xy, *gauss);
    auto W_yy = Matrix::convolve(*im_yy, *gauss);

    // Wdet = Wxx*Wyy - Wxy**2
    auto W_xy_2 = new Matrix(*W_xy);
    W_xy_2->pow(2);

    auto W_det = new Matrix(*W_xx);
    W_det->mul(*W_yy);
    W_det->sub(*W_xy_2);

    // Wtr = Wxx + Wyy
    auto W_tr = new Matrix(*W_xx);
    W_tr->add(*W_yy);

    // result = Wdet / (Wtr + 1)
    W_tr->add(1);
    auto harris = new Matrix(*W_det);
    harris->div(*W_tr);

    delete gauss;
    delete im_xx;
    delete im_xy;
    delete im_yy;
    delete W_xx;
    delete W_xy;
    delete W_yy;
    delete W_det;
    delete W_xy_2;
    delete W_tr;

    return harris;
}

Matrix* Matrix::convolve(Matrix& matrix, Matrix& kernel)
{
    return Matrix::convolve(matrix, kernel, 0.,
                            [](float acc, float mat_val, float k_val) {
                                return acc + mat_val * k_val;
                            });
}

Matrix* Matrix::convolve(Matrix& matrix, Matrix& kernel, float init_acc_value,
                         std::function<float(float, float, float)> f)
{
    Matrix* res = new Matrix(matrix.height, matrix.width);

    res->lambda([&matrix, &kernel, init_acc_value, f](size_t imgY,
                                                      size_t imgX) {
        float acc = init_acc_value;
        size_t kI = kernel.height - 1;

        int maxY = ((int)kernel.height) / 2 + kernel.height % 2;
        for (int kY = -((int)kernel.height) / 2; kY < maxY; kY++, kI--)
        {
            size_t kJ = kernel.width - 1;
            int maxX = ((int)kernel.width) / 2 + kernel.width % 2;
            for (int kX = -((int)kernel.width) / 2; kX < maxX; kX++, kJ--)
            {
                if (((int)imgY) + kY >= 0 && imgY + kY < matrix.height
                    && ((int)imgX) + kX >= 0 && imgX + kX < matrix.width)
                {
                    acc = f(acc, matrix[imgY + kY][imgX + kX], kernel[kI][kJ]);
                }
            }
        }
        return acc;
    });

    return res;
}

Tuple<Matrix, Matrix> Matrix::gauss_derivatives(Matrix* image, int kernelSize)
{
    auto gauss = Matrix::gauss_derivative_kernels(kernelSize);

    auto imx = Matrix::convolve(*image, *gauss.first);
    auto imy = Matrix::convolve(*image, *gauss.second);

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
