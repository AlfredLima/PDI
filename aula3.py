# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#%%
import numpy as np
import matplotlib.pyplot as plt
import cv2


#%%
def sumImg(img1, img2, a, b):    
    img1 = img1.astype(np.float64) 
    img2 = img2.astype(np.float64)
    img = a*img1 + b*img2
    img = np.where( img > 255.0, 255, img )    
    img = img.astype(np.uint8)
    return img

def maxImg(img1, img2):
    sizeH, sizeV, dim = img1.shape
    new_image = np.zeros([sizeH, sizeV, dim],dtype=np.uint8)
    img1 = img1.astype(np.float64) 
    img2 = img2.astype(np.float64)
    for i in range(sizeH):
        for j in range(sizeV):
            for k in range(dim):
                new_image[i][j][k] = max(img1[i][j][k],img2[i][j][k])
    return new_image

def diffAbsImg(img1, img2):
    sizeH, sizeV, dim = img1.shape
    new_image = np.zeros([sizeH, sizeV, dim],dtype=np.uint8)
    img1 = img1.astype(np.float64) 
    img2 = img2.astype(np.float64)
    for i in range(sizeH):
        for j in range(sizeV):
            for k in range(dim):
                new_image[i][j][k] = abs(img1[i][j][k]-img2[i][j][k])
    return new_image

def stretchImg(img):
    sizeH, sizeV, dim = img1.shape
    new_image = np.zeros([sizeH, sizeV, dim],dtype=np.uint8)
    img1 = img1.astype(np.float64) 
    img2 = img2.astype(np.float64)
    for i in range(sizeH):
        for j in range(sizeV):
            for k in range(dim):
                new_image[i][j][k] = abs(img1[i][j][k]-img2[i][j][k])
    return new_image

#%%%
cv2.__version__


img1 = cv2.imread('/home/alunoic/lena.png')
img2 = cv2.imread('/home/alunoic/baboon.png')

cv2.imshow('img1',img1)
cv2.waitKey(0)

cv2.imshow('img2',img2)
cv2.waitKey(0)

sizeh1, sizev1, dim1 = img1.shape
sizeh2, sizev2, dim2 = img2.shape

if sizeh1 == sizeh2 and sizev1 == sizev2 and dim1 == dim2:
    print('Mesmo tamanho')
    img3 = sumImg(img1,img2,0.5,0.5)
    cv2.imshow('img3',img3)
    cv2.waitKey(0)
    img4 = maxImg(img1,img2)
    cv2.imshow('img4',img4)
    cv2.waitKey(0)   
    img5 = maxImg(img1,img2)
    cv2.imshow('img5',img5)
    cv2.waitKey(0)
else:
    print('Diferente')

#img3 = np.zeros([SIZE,SIZE,3],dtype=np.uint8)



cv2.destroyAllWindows()
