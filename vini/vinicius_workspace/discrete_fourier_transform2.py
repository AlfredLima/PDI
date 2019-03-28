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

planes = [np.zeros(img.shape, dtype=np.float64), np.zeros(img.shape, dtype=np.float64)]
planes[0][:] = np.float64(img[:])

img2 = cv2.merge(planes)
img2 = cv2.dft(img2,flags = cv2.DFT_COMPLEX_OUTPUT)

dft_shift = np.fft.fftshift(img2)

magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))

while 0xFF & cv2.waitKey(1) != ord('q'):
    cv2.imshow('Original', img)
    cv2.imshow('Spectrum', np.uint8(magnitude_spectrum))
cv2.destroyAllWindows()