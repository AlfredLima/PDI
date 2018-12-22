import numpy as np 
import cv2
import math as m 
import sys

img = cv2.imread('/home/alfredo/Workspace/PDI/AT2/lena.png')
angle = 90

#get rotation matrix
def getRMat((cx, cy), angle, scale):
    a = scale*m.cos(angle*np.pi/180)
    b = scale*(m.sin(angle*np.pi/180))
    u = (1-a)*cx-b*cy
    v = b*cx+(1-a)*cy
    return np.array([[a,b,u], [-b,a,v]]) 

#determine shape of img
h, w = img.shape[:2]
#print h, w
#determine center of image
cx, cy = (w / 2, h / 2)

#calculate rotation matrix 
#then grab sine and cosine of the matrix
mat = getRMat((cx,cy), -int(angle), 1)
print mat
cos = np.abs(mat[0,0])
sin  = np.abs(mat[0,1])

#calculate new height and width to account for rotation
newWidth = int((h * sin) + (w * cos))
newHeight = int((h * cos) + (w * sin))
#print newWidth, newHeight

mat[0,2] += (newWidth / 2) - cx
mat[1,2] += (newHeight / 2) - cy

#this is how the image SHOULD look
dst = cv2.warpAffine(img, mat, (newWidth, newHeight))

cv2.imshow('dst', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()