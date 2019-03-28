# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 10:35:43 2019

@author: alunoic
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 10:51:40 2019

@author: alunoic
"""

import cv2
import numpy as np

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

img2 = 255 - img
ymc = cv2.split(img2)
colormap = cv2.COLORMAP_JET

while 0xFF & cv2.waitKey(1) != ord('q'):
    cv2.imshow('img', img)
    cv2.imshow('img2', img2)
    cv2.imshow('h', ymc[0])
    cv2.imshow('s', ymc[1])
    cv2.imshow('v', scaleImage2_uchar(ymc[2]))
cv2.destroyAllWindows()