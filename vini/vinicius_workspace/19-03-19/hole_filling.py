# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 10:44:11 2019

@author: alunoic
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('region-filling-reflections.tif', cv2.IMREAD_GRAYSCALE)
mask = np.zeros((img.shape[0] + 2, img.shape[1] + 2), np.uint8)

col = [0, 160, 180, 300]
row = [0, 250, 200, 240]
result = img.copy()
idx = 0
cv2.floodFill(result, mask, (row[idx], col[idx]), 255)
result_inv = cv2.bitwise_not(result)
saida = img | result_inv

plt.subplot(231), plt.imshow(img, cmap='gray'), plt.title('A')
plt.subplot(232), plt.imshow(~img, cmap='gray'), plt.title('$A^C$')
plt.subplot(233), plt.imshow(result, cmap='gray'), plt.title('$R = (X_{k-1} \oplus B')
plt.subplot(234), plt.imshow(result_inv, cmap='gray'), plt.title('$R^C$')
plt.subplot(235), plt.imshow(saida, cmap='gray')
plt.show()