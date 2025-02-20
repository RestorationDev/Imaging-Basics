# Imaging Processing Toolkit

This folder contains implementations of various image processing techniques used for enhancing and filtering images (made from scratch). Below is an overview of the methods included:

## 1. Histogram Equalization

Histogram equalization improves the contrast of an image by spreading out the most frequent intensity values. This technique is useful for enhancing details in images with poor contrast.

## 2. Median Filtering

Median filtering is a non-linear technique used to reduce noise while preserving edges. It replaces each pixel's value with the median of the intensities in its neighborhood, making it effective for removing salt-and-pepper noise.

## 3. Spatial Filtering (Blur)

Spatial filtering applies a convolution operation to smooth (blur) an image. A common implementation is Gaussian or box filtering, which helps reduce noise and minor details by averaging pixel values.

## 4. Laplacian Filtering

Laplacian filtering enhances edges by detecting regions of rapid intensity change. It uses the second derivative of the image, making it useful for edge detection and sharpening applications. By using f - gradient(f) (already handled by kernel, we can apply this filter.

## Usage

Each technique can be applied independently to images. Ensure you have the necessary dependencies installed before running the scripts.

## Dependencies

Python (>=3.x)

OpenCV (cv2)

NumPy (numpy)

