# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 09:54:50 2019

@author: alunoic
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

input_image = np.array((
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 255, 255, 255, 0, 0, 0, 255],
    [0, 255, 255, 255, 0, 0, 0, 0],
    [0, 255, 255, 255, 0, 255, 0, 0],
    [0, 0, 255, 0, 0, 0, 0, 0],
    [0, 0, 255, 0, 0, 255, 255, 0],
    [0, 255, 0, 255, 0, 0, 255, 0],
    [0, 255, 255, 255, 0, 0, 0, 0]), dtype="uint8")
    
k = np.array((
    [0, 1, 0],
    [-1, 1, 1],
    [-1, -1, 0]), dtype="int")
    
output_image = cv2.morphologyEx(input_image, cv2.MORPH_HITMISS, k)

plt.subplot(131), plt.imshow(input_image, cmap='gray'), plt.title('I')
plt.subplot(132), plt.imshow(k, cmap='gray'), plt.title('k')
plt.subplot(133), plt.imshow(output_image, cmap='gray'), plt.title('')
plt.show()