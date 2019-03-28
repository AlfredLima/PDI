# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 09:48:26 2019

@author: alunoic
"""
#%%
import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread('baboon.png', cv2.IMREAD_COLOR)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
r = np.float32([1, 0, 0])
g = np.float32([0, 1, 0])
b = np.float32([0, 0, 1])
c = np.float32([0, 1, 1])
m = np.float32([1, 0, 1])
y = np.float32([1, 1, 0])

plt.subplot('241'); plt.title('RGB');plt.imshow(img)
plt.subplot('242'); plt.title('R');plt.imshow(r*img)
plt.subplot('243'); plt.title('G');plt.imshow(g*img)
plt.subplot('244'); plt.title('B');plt.imshow(b*img)
plt.subplot('246'); plt.title('C');plt.imshow(c*img)
plt.subplot('247'); plt.title('M');plt.imshow(m*img)
plt.subplot('248'); plt.title('Y');plt.imshow(y*img)

plt.show()