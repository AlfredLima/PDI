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

def stretchImg(img, cx, cy):
    sizeH, sizeV, dim = img.shape
    new_image = np.zeros([sizeH, sizeV, dim],dtype=np.uint8)

    m = np.matrix([ [cx, 0, 0], [0, cy, 0], [0, 0, 1] ])

    for i in range(sizeH):
        for j in range(sizeV):
            point = np.matrix([i,j,1], dtype=np.uint64)
            new_point = list(map( np.uint64, point.dot(m) ))[0]
            x, y = position(new_point.item(0), new_point.item(1), sizeH, sizeV)
            new_image[x][y] = img[i][j]

    return new_image

def position(x,y, sizeH, sizeV):
    return x%sizeH , y%sizeV

def rotationPoint(point, m, img, new_img, sizeH, sizeV):
    a = list(point.dot(m))[0]
    x, y = position(a.item(0), a.item(1), sizeH, sizeV)
    new_img[x][y] = img[i][j]
    print(new_img[x][y])

def rotationImg(img, tetha):
    sizeH, sizeV, dim = img.shape
    new_image = np.zeros([sizeH, sizeV, dim],dtype=np.uint8)
    
    co = np.cos( tetha * np.pi/180 )
    si = np.sin( tetha * np.pi/180 )
    m = np.matrix([ [co, -si, 0], [si, co, 0], [0, 0, 1] ])
    
    for i in range(sizeH):
        for j in range(sizeV):
            point = np.matrix([i,j,0], dtype=np.float64)
            new_point = list( map(np.int32, point.dot(m)) )[0]
            x, y = new_point.item(0), new_point.item(1)
            
            try:

                if x < 0  or y < 0:
                    continue
                new_image[x][y] = img[i][j]
                
            except Exception as e:
                pass    
    
    return new_image


img1 = cv2.imread('/home/alfredo/Workspace/PDI/AT2/lena.png')
img2 = cv2.imread('/home/alfredo/Workspace/PDI/AT2/baboon.png')

show = 1

if show :
    cv2.imshow('img1',img1)
    cv2.waitKey(0)

    #cv2.imshow('img2',img2)
    #cv2.waitKey(0)

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
    #img6 = rotationImg(img1,90)
    #cv2.imshow('img6',img6)
    #cv2.waitKey(0)
    img7 = stretchImg(img1,2,2)
    cv2.imshow('img7',img7)
    cv2.waitKey(0)
else:
    print('Diferente')

cv2.destroyAllWindows()
