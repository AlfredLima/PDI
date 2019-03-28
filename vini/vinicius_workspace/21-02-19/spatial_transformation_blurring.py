# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 10:39:59 2019

@author: alunoic
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

def doNothing(param):
    pass


#%%

img = cv2.imread('baboon.png', cv2.IMREAD_COLOR)

xsize = 3
ysize = 3
cv2.namedWindow('img', cv2.WINDOW_KEEPRATIO)
cv2.namedWindow('img2', cv2.WINDOW_KEEPRATIO)

cv2.createTrackbar('xsize', 'img2', xsize, 50, doNothing)
cv2.createTrackbar('ysize', 'img2', ysize, 50, doNothing)

while cv2.waitKey(1) != ord('q'):
    xsize = cv2.getTrackbarPos('xsize', 'img2')
    ysize = cv2.getTrackbarPos('ysize', 'img2')
    
    img2 = cv2.blur(img, (xsize+1, ysize+1))
    cv2.imshow('img', img)
    cv2.imshow('img2', img2)
cv2.destroyAllWindows()