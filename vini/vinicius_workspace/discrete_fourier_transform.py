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
    

img = cv2.imread('rectangle.jpg', cv2.IMREAD_GRAYSCALE)
cv2.namedWindow('Original', cv2.WINDOW_KEEPRATIO)
cv2.namedWindow('Plane 0 - Real', cv2.WINDOW_KEEPRATIO)
cv2.namedWindow('Plane 1 - Imaginary', cv2.WINDOW_KEEPRATIO)

planes = [np.zeros(img.shape, dtype=np.float64), np.zeros(img.shape, dtype=np.float64)]
planes[0][:] = np.float64(img[:])

img2 = cv2.merge(planes)
img2 = cv2.dft(img2)

planes = cv2.split(img2)

cv2.normalize(planes[0], planes[0], 1, 0, cv2.NORM_MINMAX)
cv2.normalize(planes[1], planes[1], 1, 0, cv2.NORM_MINMAX)

while 0xFF & cv2.waitKey(1) != ord('q'):
    cv2.imshow('Original', img)
    cv2.imshow('Plane 0 - Real', planes[0])
    cv2.imshow('Plane 1 - Imaginary', planes[1])
cv2.destroyAllWindows()