# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 10:00:03 2019

@author: alunoic
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Feb  5 10:37:18 2019

@author: alunoic
"""

import cv2
import numpy as np


img = cv2.imread('rectangle.jpg', cv2.IMREAD_GRAYSCALE)
cv2.namedWindow('Original', cv2.WINDOW_KEEPRATIO)
cv2.namedWindow('Spectrum', cv2.WINDOW_KEEPRATIO)
cv2.namedWindow('Spectrum Shift', cv2.WINDOW_KEEPRATIO)

planes = [np.zeros(img.shape, dtype=np.float64), 
          np.zeros(img.shape, dtype=np.float64)]
          
planes[0][:] = np.float64(img[:])
planes[1][:] = np.float64(img[:])
cv2.normalize(planes[0], planes[0], 1, 0, cv2.NORM_MINMAX)
cv2.normalize(planes[0], planes[1], 1, 0, cv2.NORM_MINMAX)

img2 = cv2.merge(planes)
img2 = cv2.dft(img2)
planes = cv2.split(img2)

magnitude_spectrum = cv2.magnitude(planes[0],planes[1])
magnitude_spectrum += 1
magnitude_spectrum = np.log(magnitude_spectrum)

dft_shift = np.fft.fftshift(magnitude_spectrum)

cv2.normalize(magnitude_spectrum, magnitude_spectrum, 1, 0, cv2.NORM_MINMAX)
cv2.normalize(dft_shift, dft_shift, 1, 0, cv2.NORM_MINMAX)

while 0xFF & cv2.waitKey(1) != ord('q'):
    cv2.imshow('Original', img)
    cv2.imshow('Spectrum', magnitude_spectrum)
    cv2.imshow('Spectrum Shift', dft_shift)
cv2.destroyAllWindows()