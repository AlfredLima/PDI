# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 09:48:26 2019

@author: alunoic
"""
#%%
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('rgbcube_kBKG.png', cv2.IMREAD_COLOR)
bgr = cv2.split(img)
colormap = 1


plt.subplot('221'); plt.title('B');plt.imshow(cv2.applyColorMap(bgr[0], colormap))
plt.subplot('222'); plt.title('G');plt.imshow(cv2.applyColorMap(bgr[1], colormap))
plt.subplot('223'); plt.title('R');plt.imshow(cv2.applyColorMap(bgr[2], colormap))
plt.subplot('224'); plt.title('Original');plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))



plt.show()