#include "include/morphology.hh"

#include <limits>

Matrix* ellipse_kernel(size_t width, size_t height)
{
    auto f_ = [](int i, int a) {
        return std::pow(i - a / 2, 2) / std::pow(a / 2, 2);
    };

    auto kernel = new Matrix(width, height);
    kernel->lambda([&f_, width, height](size_t i, size_t j) {
        return f_(i, height) + f_(j, width) <= 1 ? 1. : 0.;
    });
    return kernel;
}

Matrix* rectangle_kernel(size_t width, size_t height)
{
    auto kernel = new Matrix(width, height);
    kernel->fill(1.0);
    return kernel;
}

Matrix* dilation(Matrix& matrix, Matrix& kernel)
{
    return Matrix::convolve(matrix, kernel, std::numeric_limits<float>::min(),
                            [](float acc, float mat_val, float k_val) {
                                return k_val > 0.00001 ? std::max(acc, mat_val)
                                                       : acc;
                            });
}

Matrix* erosion(Matrix& matrix, Matrix& kernel)
{
    return Matrix::convolve(matrix, kernel, std::numeric_limits<float>::max(),
                            [](float acc, float mat_val, float k_val) {
                                return k_val > 0.00001 ? std::min(acc, mat_val)
                                                       : acc;
                            });
}

Matrix* opening(Matrix& matrix, Matrix& kernel)
{
    auto ero = erosion(matrix, kernel);
    auto dil = dilation(*ero, kernel);
    delete ero;
    return dil;
}

Matrix* closing(Matrix& matrix, Matrix& kernel)
{
    auto dil = dilation(matrix, kernel);
    auto ero = erosion(*dil, kernel);
    delete dil;
    return ero;
}

Matrix* eroded_mask(Matrix& grayscale_image, size_t border)
{
    auto mask = new Matrix(grayscale_image);
    mask->lambda([](float e) { return e > 0.00001 ? 1. : 0.; });

    auto rect_kernel = rectangle_kernel(3, 3);
    auto mask_er_tmp = closing(*mask, *rect_kernel);

    auto ell_kernel = ellipse_kernel(border * 2, border * 2);
    auto mask_er = erosion(*mask_er_tmp, *ell_kernel);

    // bubble2maskeroded =
    mask_er->lambda([](float e) { return e > 0.00001 ? 1. : 0.; });

    delete mask;
    delete rect_kernel;
    delete mask_er_tmp;
    delete ell_kernel;

    return mask_er;
}

Matrix* harris_points(Matrix& harrisim)
{
    auto ell_kernel = ellipse_kernel(20, 20);
    auto dil = dilation(harrisim, *ell_kernel);

    auto is_close = dil->is_close(harrisim);
    auto mask = eroded_mask(harrisim, 10);

    auto detection = new Matrix(harrisim);

    auto min = detection->min();
    auto max = detection->max();

    auto lambda_ = [max, min](float a) {
        return a > min + 0.5 * (max - min) ? 1 : 0;
    };
    detection->lambda(lambda_);
    detection->mul(*is_close);
    detection->mul(*mask);

    delete is_close;
    delete dil;
    delete ell_kernel;
    delete mask;

    return detection;
}
