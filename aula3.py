# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import cv2
import os,sys

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

def norma(x, min_x, max_x):
    return (x-min_x)/(max_x-min_x)

def getRMat((cx, cy), angle, scale):
    a = scale*m.cos(angle*np.pi/180)
    b = scale*(m.sin(angle*np.pi/180))
    u = (1-a)*cx-b*cy
    v = b*cx+(1-a)*cy
    return np.array([[a,b,u], [-b,a,v]]) 


def rotationImg(img, tetha):
    sizeH, sizeV, dim = img1.shape
    new_image = np.zeros([sizeH, sizeV, dim],dtype=np.float64)
    img = img.astype(np.float64) 
    
    co = np.cos( tetha * np.pi/180 )
    si = np.sin( tetha * np.pi/180 )
    matrix_rotation = np.matrix( [ [co, -si, 0] , [si, co, 0], [0, 0, 1] ] )

    for i in range(sizeH):
        for j in range(sizeV):
            new_image[i][j] = np.matmul(matrix_rotation, img[i][j])

    max_x, min_x = np.max(new_image), np.min(new_image)
    
    for i in range(sizeH):
        for j in range(sizeV):
            new_image[i][j] = list(map(lambda x: x(new_image[i][j], min_x, max_x), [norma]))[0]

    new_image = (255 * new_image).astype(np.uint8)
    #print(new_image)
    return new_image


img1 = cv2.imread('/home/alfredo/Workspace/PDI/AT2/lena.png')
img2 = cv2.imread('/home/alfredo/Workspace/PDI/AT2/baboon.png')

cv2.imshow('img1',img1)
cv2.waitKey(0)

cv2.imshow('img2',img2)
cv2.waitKey(0)

sizeh1, sizev1, dim1 = img1.shape
sizeh2, sizev2, dim2 = img2.shape

if sizeh1 == sizeh2 and sizev1 == sizev2 and dim1 == dim2:
    print('Mesmo tamanho')
    #img3 = sumImg(img1,img2,0.1,0.1)
    #cv2.imshow('img3',img3)
    #cv2.waitKey(0)
    #img4 = maxImg(img1,img2)
    #cv2.imshow('img4',img4)
    #cv2.waitKey(0)   
    #img5 = diffAbsImg(img1,img2)
    #cv2.imshow('img5',img5)
    #cv2.waitKey(0)
    img6 = rotationImg(img1,90)
    cv2.imshow('img6',img6)
    cv2.waitKey(0)
else:
    print('Diferente')

cv2.destroyAllWindows()
