# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 09:48:26 2019

@author: alunoic
"""
#%%
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('baboon.png', cv2.IMREAD_COLOR)
bgr = cv2.split(img)

plt.subplot('221'); plt.title('B');
plt.imshow(bgr[0], 'gray')
plt.subplot('222'); plt.title('G');
plt.imshow(bgr[1], 'gray')
plt.subplot('223'); plt.title('R');
plt.imshow(bgr[2], 'gray')
plt.subplot('224'); plt.title('Original');
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()