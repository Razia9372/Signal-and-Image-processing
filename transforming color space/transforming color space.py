import numpy as np
from skimage.io import imread
from skimage.color import rgb2lab, lab2rgb
import matplotlib.pylab as plt
im = imread('dataset/Surface_Laptop_3_04.jpg')
im1 = rgb2lab(im)
im1[...,1] = im1[...,2] = 0
im1 = lab2rgb(im1)
plt.figure(figsize=(20,10))
plt.subplot(121), plt.imshow(im), plt.axis('off'),
plt.title('Original image', size=20)
plt.subplot(122), plt.imshow(im1), plt.axis('off'),
plt.title('Gray scale image', size=20)
plt.show()