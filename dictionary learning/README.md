# Dictionary Learning for Image Reconstruction

## Overview

This experiment explores dictionary learning and sparse coding for reconstructing image regions affected by Gaussian noise.

The implementation uses Python and scikit-learn to learn a dictionary of image patches and reconstruct them using Orthogonal Matching Pursuit (OMP).

## Methodology

### 1. Image Preprocessing

- Load an input image and convert it to grayscale.
- Introduce Gaussian noise into the lower half of the image.
- Extract overlapping 7 × 7 patches from the noisy region.

### 2. Dictionary Learning

A dictionary containing 256 components is learned using `MiniBatchDictionaryLearning`.

The goal is to represent image patches using combinations of learned dictionary elements.

### 3. Sparse Reconstruction

Orthogonal Matching Pursuit is used to obtain sparse representations with at most two nonzero coefficients per patch.

The reconstructed patches are combined to form the reconstructed lower half of the image.

The upper half remains unchanged.

### 4. Result Visualization

The experiment displays:

- The noise-corrupted image.
- The reconstructed image.
- Difference images relative to the original image.
- A numerical difference norm.

## Main Concepts

- Sparse representation
- Dictionary learning
- Orthogonal Matching Pursuit
- Image patch extraction
- Image reconstruction
- Gaussian noise

## Technologies

Python, NumPy, scikit-image, scikit-learn, and Matplotlib.

## Limitations

This is an educational experiment rather than a validated image denoising benchmark.

The dictionary is learned from noisy image patches, and the reconstruction uses the original upper half of the image. The reported difference norm is not a standardized quality metric such as PSNR or SSIM.

The original script also contains a local image path and may require library-version adjustments before running on another computer.
