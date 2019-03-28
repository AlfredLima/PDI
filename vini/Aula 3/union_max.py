# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 09:52:19 2018

@author: alunoic
"""

import cv2
import numpy as np
#%%
im1 = cv2.imread("lena.png", cv2.IMREAD_GRAYSCALE)
im2 = cv2.imread("baboon.png", cv2.IMREAD_GRAYSCALE)

x_range, y_range = im1.shape

im_res = np.zeros([x_range, y_range], dtype=np.uint8)


#%% LOOP
for i in range(x_range):
    for j in range(y_range):
        im_res[i][j] = max(im1[i][j],im2[i][j])
        
#%%

#cv2.bitwise_and(im1,im2)
cv2.namedWindow('RESULT', cv2.WINDOW_KEEPRATIO)
cv2.imshow('RESULT', im_res)
cv2.waitKey(0)
cv2.destroyAllWindows()