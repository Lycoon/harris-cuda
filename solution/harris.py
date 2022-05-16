import sys
from typing import Tuple

import cv2
import numpy as np
from scipy import signal


def gauss_kernel(size: int, sizey: int=None) -> np.array:
    """
    Returns a 2D Gaussian kernel for convolutions.
    
    Parameters
    ----------
    size: int
        Size of the kernel to build
    
    Returns
    -------
    kernel: np.array of shape (size, size) and dtype np.float32
        Resulting Gaussian kernel where kernel[i,j] = Gaussian(i, j, mu=(0,0), sigma=(size/3, size/3))
    """
    size = int(size)
    sizey = int(sizey) if sizey is not None else size
    y, x = np.mgrid[-size:size+1, -sizey:sizey+1]

    # x and y coefficients of a 2D gaussian with standard dev half of size
    # (ignore scale factor)
    g = np.exp(-(x**2/(2*(0.33*size)**2)+y**2/(2*(0.33*sizey)**2)))
    return g


def gauss_derivative_kernels(size: int, sizey: int=None) -> Tuple[np.array, np.array]:
    """
    Returns two 2D Gaussian derivative kernels (x and y) for convolutions.
    
    Parameters
    ----------
    size: int
        Size of the kernels to build
    
    Returns
    -------
    (gx, gy): tupe of (np.array, np.array), each of shape (size, size) and dtype np.float32
        Resulting Gaussian kernels where kernel[i,j] = Gaussian_z(i, j, mu=(0,0), sigma=(size/3, size/3))
        where Gaussian_z is either the x or the y Gaussian derivative.
    """
    size = int(size)
    sizey = int(sizey) if sizey is not None else size
    y, x = np.mgrid[-size:size+1, -sizey:sizey+1]

    #x and y derivatives of a 2D gaussian with standard dev half of size
    # (ignore scale factor)
    gx = - x * np.exp(-(x**2/(2*(0.33*size)**2)+y**2/(2*(0.33*sizey)**2)))
    gy = - y * np.exp(-(x**2/(2*(0.33*size)**2)+y**2/(2*(0.33*sizey)**2)))

    return gx,gy


def gauss_derivatives(im: np.array, size: int, sizey: int=None) -> Tuple[np.array, np.array]:
    """
    Returns x and y gaussian derivatives for a given image.
    
    Parameters
    ----------
    im: np.array of shape (rows, cols)
        Input image
    size: int
        Size of the kernels to use
    
    Returns
    -------
    (Ix, Iy): tupe of (np.array, np.array), each of shape (rows, cols)
        Derivatives (x and y) of the image computed using Gaussian derivatives (with kernel of size `size`).
    """
    gx,gy = gauss_derivative_kernels(size, sizey=sizey)

    imx = signal.convolve(im, gx, mode='same')
    imy = signal.convolve(im, gy, mode='same')

    return imx,imy


def compute_harris_response(image):  #, k=0.05):
    """
    Returns the Harris cornerness response of a given image.
    
    Parameters
    ----------
    im: np.array of shape (rows, cols)
        Input image
    
    Returns
    -------
    response: np.array of shape (rows, cols) and dtype np.float32
        Harris cornerness response image.
    """
    DERIVATIVE_KERNEL_SIZE = 3
    OPENING_SIZE = 3
    
    #derivatives
    imx,imy = gauss_derivatives(image, DERIVATIVE_KERNEL_SIZE)

    #kernel for weighted sum
    gauss = gauss_kernel(OPENING_SIZE) # opening param

    #compute components of the structure tensor
    Wxx = signal.convolve(imx*imx,gauss, mode='same')
    Wxy = signal.convolve(imx*imy,gauss, mode='same')
    Wyy = signal.convolve(imy*imy,gauss, mode='same')

    #determinant and trace
    Wdet = Wxx*Wyy - Wxy**2
    Wtr = Wxx + Wyy
#     print(Wdet.min(), Wdet.max(), Wdet.mean())
#     print(Wtr.min(), Wtr.max(), Wtr.mean())

    # return Wdet - k * Wtr**2 # k is hard to tune
    # return Wdet / Wtr # we would need to filter NaNs
    return Wdet / (Wtr + 1)  # 1 seems to be a reasonable value for epsilon


# RUN ME
# mathematical morphology magic: this returns an eroded (shrunk) mask
def bubble2maskeroded(img_gray: np.array, border: int=10) -> np.array:
    """
    Returns the eroded mask of a given image, to remove pixels which are close to the border.
    
    Parameters
    ----------
    im: np.array of shape (rows, cols)
        Input image
    
    Returns
    -------
    mask: np.array of shape (rows, cols) and dtype bool
        Image mask.
    """
    if img_gray.ndim > 2:
        raise ValueError(
            """bubble2maskeroded: img_gray must be a grayscale image.
            The image you passed has %d dimensions instead of 2.
            Try to convert it to grayscale before passing it to bubble2maskeroded.
            """ % (img_gray.ndim, ))
    mask = img_gray > 0
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (border*2,border*2))
    # new: added a little closing below because some bubbles have some black pixels inside
    mask_er = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3,3)))
    mask_er = cv2.erode(mask.astype(np.uint8), 
                        kernel, 
                        borderType=cv2.BORDER_CONSTANT, 
                        borderValue=0)
    return mask_er > 0


def detect_harris_points(image_gray: np.array, max_keypoints: int=30, 
                         min_distance: int=25, threshold: float=0.1) -> np.array:
    """
    Detects and returns a sorted list of coordinates for each corner keypoint detected in an image.
    
    Parameters
    ----------
    image_gray: np.array
        Input image
    max_keypoints: int, default=30
        Number of keypoints to return, at most (we may have less keypoints)
    min_distance: int, default=25
        Minimum distance between two keypoints
    threshold: float, default=0.1
        For each keypoint k_i, we ensure that its response h_i will verify
        $h_i > min(response) + threshold * (max(reponse) - min(response))$
    
    Returns
    -------
    corner_coord: np.array of shape (N, 2) and dtype int
        Array of corner keypoint 2D coordinates, with N <= max_keypoints
    """
    # 1. Compute Harris corner response
    harris_resp = compute_harris_response(image_gray)
    
    # 2. Filtering
    # 2.0 Mask init: all our filtering is performed using a mask
    detect_mask = np.ones(harris_resp.shape, dtype=bool)
    # 2.1 Background and border removal
    detect_mask &= bubble2maskeroded(image_gray, border=min_distance)
    # 2.2 Response threshold
    detect_mask &= harris_resp > harris_resp.min()+threshold*(harris_resp.max()-harris_resp.min())
    # 2.3 Non-maximal suppression
    dil = cv2.dilate(harris_resp, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (min_distance, min_distance)))
    detect_mask &= np.isclose(dil, harris_resp)  # keep only local maximas
               
    # 3. Select, sort and filter candidates
    # get coordinates of candidates
    candidates_coords = np.transpose(detect_mask.nonzero())
    # ...and their values
    candidate_values = harris_resp[detect_mask]
    #sort candidates
    sorted_indices = np.argsort(candidate_values)
    # keep only the bests
    best_corners_coordinates = candidates_coords[sorted_indices][:max_keypoints]

    return best_corners_coordinates


def best_points(image_path: str):

    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    harris = compute_harris_response(gray)
    coords = detect_harris_points(harris, max_keypoints=-1, min_distance=20)

    output = ""

    for p in coords:
        output += f"x: {p[1]} | y: {p[0]}\n"
        cv2.circle(image, (p[1], p[0]), 3, (0, 255, 0), -1)
    
    with open("output.txt", "w") as f:
        f.write(output)
    
    print(coords)
    cv2.imwrite("output.png", image)


if __name__ == "__main__":
    
    if len(sys.argv) != 2:
        print("Usage: ./harris <image>")
        exit()

    best_points(sys.argv[1])
