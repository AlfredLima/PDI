# -*- coding: utf-8 -*-
"""
Created on Tue Feb  5 10:08:19 2019

@author: alunoic
"""

import cv2
import numpy as np

def createWhiteDisk(height = 100, width = 100, xc = 50, yc = 50, rc = 20):
    disk = np.zeros((height, width), np.float64)
    for x in range(disk.shape[0]):
        for y in range(disk.shape[1]):
            if (x - xc)*(x - xc)+(y - yc)*(y - yc) <= rc * rc:
                disk[x][y] = 1.0
    return disk
    
def apply_log_transform(img):
    img2 = np.copy(img)
    img2 += 1
    img2 = np.log(img2)
    return img2
    
cv2.namedWindow('white_disk', cv2.WINDOW_KEEPRATIO)
cv2.imshow('white_disk', apply_log_transform(createWhiteDisk(100,100,10,10,10)))
cv2.waitKey(0)
cv2.destroyAllWindows()