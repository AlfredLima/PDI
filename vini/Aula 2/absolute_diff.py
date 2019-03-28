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

im1 = np.float32(im1)
im2 = np.float32(im2)

im_res = im1 - im2
im_res = np.where(im_res<0,-im_res,im_res)
im_res = im_res/255

#%%
cv2.namedWindow('img', cv2.WINDOW_KEEPRATIO)
cv2.imshow('img', im_res)
cv2.waitKey(0)
cv2.destroyAllWindows()