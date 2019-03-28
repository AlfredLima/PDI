# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 09:52:19 2018

@author: alunoic
"""

import cv2
import numpy as np
#%%
im1 = cv2.imread("square1.png", cv2.IMREAD_GRAYSCALE)
im2 = cv2.imread("square2.png", cv2.IMREAD_GRAYSCALE)

x_range, y_range = im1.shape

im_and = np.zeros([x_range, y_range], dtype=np.uint8)
im_or = np.zeros([x_range, y_range], dtype=np.uint8)
im_not = np.zeros([x_range, y_range], dtype=np.uint8)
im_xor = np.zeros([x_range, y_range], dtype=np.uint8)


#%% LOOP
for i in range(x_range):
    for j in range(y_range):
        im_and[i][j] = im1[i][j] & im2[i][j]
        im_or[i][j] = im1[i][j] | im2[i][j]
        im_not[i][j] = ~im1[i][j]
        im_xor[i][j] = im1[i][j] ^ im2[i][j]
        
#%%

#cv2.bitwise_and(im1,im2)
cv2.namedWindow('AND', cv2.WINDOW_KEEPRATIO)
cv2.imshow('AND', im_and)
cv2.namedWindow('OR', cv2.WINDOW_KEEPRATIO)
cv2.imshow('OR', im_or)
cv2.namedWindow('NOT', cv2.WINDOW_KEEPRATIO)
cv2.imshow('NOT', im_not)
cv2.namedWindow('XOR', cv2.WINDOW_KEEPRATIO)
cv2.imshow('XOR', im_xor)
cv2.waitKey(0)
cv2.destroyAllWindows()