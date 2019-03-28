# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 09:52:19 2018

@author: alunoic
"""

import cv2
import numpy as np


#%%
im1 = cv2.imread("hat.jpeg", cv2.IMREAD_GRAYSCALE)
im2 = cv2.imread("no_hat.jpeg", cv2.IMREAD_GRAYSCALE)

im_res = np.float32(im1) - np.float32(im2)
im_res = np.where(im_res<0,-im_res,im_res)
im_res = np.where(im_res<70,0,255)
im_res = im_res/255

#%%
cv2.namedWindow('img1', cv2.WINDOW_KEEPRATIO)
cv2.imshow('img1', im1)
cv2.namedWindow('img2', cv2.WINDOW_KEEPRATIO)
cv2.imshow('img2', im2)
cv2.namedWindow('diff', cv2.WINDOW_KEEPRATIO)
cv2.imshow('diff', im_res)
cv2.waitKey(0)
cv2.destroyAllWindows()