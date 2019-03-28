# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 10:39:59 2019

@author: alunoic
"""

import cv2

def doNothing(param):
    pass


#%%

img = cv2.imread('baboon.png', cv2.IMREAD_COLOR)

wsize = 3

cv2.namedWindow('img', cv2.WINDOW_KEEPRATIO)
cv2.namedWindow('img2', cv2.WINDOW_KEEPRATIO)

cv2.createTrackbar('wsize', 'img2', wsize, 10, doNothing)

while cv2.waitKey(1) != ord('q'):
    wsize = cv2.getTrackbarPos('wsize', 'img2')
    
    img2 = cv2.Laplacian(img, cv2.CV_16S, ksize=2*wsize + 1, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
    cv2.imshow('img', img)
    cv2.imshow('img2', img2)
cv2.destroyAllWindows()