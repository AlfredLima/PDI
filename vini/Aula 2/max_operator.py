# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 09:52:19 2018

@author: alunoic
"""

import cv2
import numpy as np


#%%
im1 = cv2.imread("lena.png")
im2 = cv2.imread("baboon.png")

x_range, y_range, z_range = im1.shape

im_res = np.zeros([x_range, y_range, z_range], dtype=np.uint8)

for i in range(x_range):
    for j in range(y_range):
        for k in range(z_range):
            im_res[i][j][k] = max(im1[i][j][k], im2[i][j][k])


#%%
cv2.namedWindow('img', cv2.WINDOW_KEEPRATIO)
cv2.imshow('img', im_res)
cv2.waitKey(0)
cv2.destroyAllWindows()