# Signal and Image Processing

A collection of Python-based experiments exploring fundamental techniques in digital image processing, sparse representation, image reconstruction, and color space transformations.

The repository contains implementations and demonstrations of image processing methods using Python, NumPy, scikit-image, scikit-learn, and Matplotlib.

## Projects

### 1. Dictionary Learning for Image Reconstruction

This experiment investigates sparse image representation and reconstruction using a learned dictionary.

**Methods and techniques:**
- Extraction of 7 × 7 image patches
- Addition of Gaussian noise to part of an input image
- Dictionary learning using `MiniBatchDictionaryLearning`
- Sparse coding using Orthogonal Matching Pursuit (OMP)
- Image reconstruction from overlapping patches
- Visual comparison of reconstructed images and reconstruction errors

**Implementation details:**
- Dictionary size: 256 components
- Sparse reconstruction: 2 nonzero coefficients per patch
- Dictionary training and reconstruction performed on the noisy image region

The experiment demonstrates how learned image features can be combined through sparse coding to reconstruct image patches.

The implementation uses the original image as a reference for visual error analysis. It is an educational reconstruction experiment rather than a validated denoising benchmark.

### 2. Color Space Transformations

This project explores image representation and manipulation in the CIELAB color space.

Images are converted from RGB to Lab, where the components represent:

- **L\***: Lightness
- **a\***: Green–red color dimension
- **b\***: Blue–yellow color dimension

The experiments modify individual channels and transform the results back to RGB to visualize their effects.

The repository also demonstrates grayscale conversion by removing the chromatic components.

### 3. Image Inversion

Additional Jupyter notebook experiments explore image inversion and transformations.

These exercises complement the color space experiments and provide practical examples of pixel-level image manipulation.

## Technologies

- Python
- NumPy
- scikit-image
- scikit-learn
- Matplotlib
- Jupyter Notebook

## Learning Objectives

These projects explore several foundational concepts:

- Digital image representation and processing
- Sparse coding and dictionary learning
- Image reconstruction
- Color space transformations
- Numerical computation and visualization

## Reproducibility Notes

The experiments were developed as educational exercises. Some scripts use local file paths and may require adjustments to run in a different environment.

Library compatibility may also depend on the versions of Python and the scientific computing packages installed.

## Project Scope

This repository documents practical image processing exercises and numerical experiments. The results should be interpreted as demonstrations of the implemented methods rather than production-ready or independently benchmarked image processing systems.
