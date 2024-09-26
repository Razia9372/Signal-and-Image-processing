import numpy as np
from skimage.io import imread
from skimage.color import rgb2lab, lab2rgb
import matplotlib.pylab as plt
im = imread('dataset/transforming color space/Surface_Laptop_3_04.jpg')
# when luminacity is zero
im1 = rgb2lab(im)
im1[...,0] = 0
im1 = lab2rgb(im1)
plt.figure(figsize=(20,10))
plt.subplot(121), plt.imshow(im), plt.axis('off'),
plt.title('Original image', size=20)
plt.subplot(122), plt.imshow(im1), plt.axis('off'),
plt.title('Gray scale image(luminacity=0)', size=20)
plt.show()
# when green-red channel is zero
im2 = rgb2lab(im)
im2[...,1] = 0
im2 = lab2rgb(im2)
plt.figure(figsize=(20,10))
plt.subplot(121), plt.imshow(im), plt.axis('off'),
plt.title('Original image', size=20)
plt.subplot(122), plt.imshow(im2), plt.axis('off'),
plt.title('Gray scale image(green-red=0)', size=20)
plt.show()
# when blue-yellow channel is zero
im3 = rgb2lab(im)
im3[...,2] = 0
im3 = lab2rgb(im3)
plt.figure(figsize=(20,10))
plt.subplot(121), plt.imshow(im), plt.axis('off'),
plt.title('Original image', size=20)
plt.subplot(122), plt.imshow(im3), plt.axis('off'),
plt.title('Gray scale image(blue-yellow=0)', size=20)
plt.show()