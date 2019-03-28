# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 10:51:40 2019

@author: alunoic
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

#%%

def scaleImage2_uchar(src):
    tmp = np.copy(src)
    if src.dtype != np.float32:
        tmp = np.float32(tmp)
    cv2.normalize(tmp, tmp, 1, 0, cv2.NORM_MINMAX)
    tmp = 255 * tmp
    tmp = np.uint8(tmp)
    return tmp

img = cv2.imread('chips.png', cv2.IMREAD_COLOR)

img2 = cv2.cvtColor(img, cv2.COLOR_BGR2HSV_FULL)
hsv = cv2.split(img2)


#histr = cv2.calcHist([img2], [0], None, [256], [0, 256])
#plt.plot(histr)
#plt.xlim([0, 256])
#plt.show()

#for i, col in enumerate(color):
#     histr = cv2.calcHist([img], [i], None, [256], [0, 256])
#     plt.plot(histr, color=col)
#     plt.xlim([0, 256])
# plt.show()


#for i in range(391):
#    for j in range(518):
#        for k in range(3):
#            img2[i][j][k] = img2[i][j][k] * hsv[1][i][j]
#            
#cv2.normalize(img2, img2, 255, 0, cv2.NORM_MINMAX)

while 0xFF & cv2.waitKey(1) != ord('q'):
    cv2.imshow('img', img)
    cv2.imshow('img2', img2)
    cv2.imshow('h', hsv[0])
    cv2.imshow('s', hsv[1])
    cv2.imshow('v', scaleImage2_uchar(hsv[2]))

cv2.destroyAllWindows()