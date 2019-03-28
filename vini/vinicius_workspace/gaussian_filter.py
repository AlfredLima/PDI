# -*- coding: utf-8 -*-
"""
Created on Tue Feb  5 10:37:18 2019

@author: alunoic
"""

import cv2
import numpy as np


def doNothing(param):
    pass

def scaleImage2_uchar(src):
    tmp = np.copy(src)
    if src.dtype != np.float32:
        tmp = np.float32(tmp)
    cv2.normalize(tmp, tmp, 1, 0, cv2.NORM_MINMAX)
    tmp = 255 * tmp
    tmp = np.uint8(tmp)
    return tmp

def create2DGaussian(rows = 100, cols = 100, mx = 50, my = 50, sx = 10, sy = 100, theta = 0):
    xx0, yy0 = np.meshgrid(range(rows), range(cols))
    xx0 -= mx
    yy0 -= my
    theta = np.deg2rad(theta)
    xx = xx0 * np.cos(theta) - yy0 * np.sin(theta)
    yy = xx0 * np.sin(theta) + yy0 * np.cos(theta)
    try:
        img = np.exp( - ((xx**2)/(2*sx**2) + (yy**2)/(2*sy**2)))
    except ZeroDivisionError:
        img = np.zeros((rows, cols), dtype='float64')
    cv2.normalize(img, img, 1, 0, cv2.NORM_MINMAX)
    return img
    

rows = 100
cols = 100
theta = 0
xc = 50
yc = 50
sx = 30
sy = 10
theta = 0

cv2.namedWindow('img', cv2.WINDOW_KEEPRATIO)
cv2.createTrackbar('xc', 'img', xc, int(rows), doNothing)
cv2.createTrackbar('yc', 'img', yc, int(cols), doNothing)
cv2.createTrackbar('sx', 'img', sx, int(rows), doNothing)
cv2.createTrackbar('sy', 'img', sy, int(cols), doNothing)
cv2.createTrackbar('theta', 'img', theta, 360, doNothing)

while 0xFF & cv2.waitKey(1) != ord('q'):
    xc = cv2.getTrackbarPos('xc', 'img')
    yc = cv2.getTrackbarPos('yc', 'img')
    sx = cv2.getTrackbarPos('sx', 'img')
    sy = cv2.getTrackbarPos('sy', 'img')
    theta = cv2.getTrackbarPos('theta', 'img')
    img = create2DGaussian(rows, cols, xc, yc, sx, sy, theta)
    cv2.imshow('img', cv2.applyColorMap(scaleImage2_uchar(img), cv2.COLORMAP_JET))
cv2.destroyAllWindows()
