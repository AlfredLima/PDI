# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 09:49:00 2019

@author: alunoic
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

noisy = cv2.imread('noisy-fingerprint.tif', cv2.IMREAD_GRAYSCALE)
kernel = np.ones((3,3), np.uint8)
noisy_o = cv2.morphologyEx(noisy, cv2.MORPH_OPEN, kernel)
noisy_oc = cv2.morphologyEx(noisy, cv2.MORPH_OPEN, kernel)

plt.subplot(131), plt.imshow(noisy, cmap='gray'), plt.title('Original')
plt.subplot(132), plt.imshow(noisy_o, cmap='gray'), plt.title('Opened')
plt.subplot(133), plt.imshow(noisy_oc, cmap='gray'), plt.title('Opened and closed')
plt.show()