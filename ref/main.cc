#include <iostream>

#include "src/include/matrix.hh"
#include "src/include/png.hh"

int main()
{
    std::cout << "Harris corner detector"
              << "\n";

    const auto DERIVATIVE_KERNEL_SIZE = 3;
    const auto OPENING_SIZE = 3;

    auto image = ImagePNG::read((char*)"../../twin_it/bubbles_200dpi/b003.png");
    auto gray = image->grayscale_matrix();
    ImagePNG::write_matrix("gray.png", gray);

    auto derivatives = Matrix::gauss_derivatives(gray, DERIVATIVE_KERNEL_SIZE);

    ImagePNG::write_matrix("gauss_x.png", derivatives.first);
    ImagePNG::write_matrix("gauss_y.png", derivatives.second);

    auto gauss = Matrix::gauss_kernel(OPENING_SIZE);

    auto im_xx = new Matrix(*(derivatives.first));
    im_xx->mul(*(derivatives.first));
    auto im_xy = new Matrix(*(derivatives.first));
    im_xy->mul(*(derivatives.second));
    auto im_yy = new Matrix(*(derivatives.second));
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

    ImagePNG::write_matrix("harris.png", harris);

    delete image;
    delete gray;
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
    delete harris;
}
