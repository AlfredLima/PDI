import numpy as np
import cv2
GRAY = 32

def tomGray(pixel):
	return np.dot(pixel, [0.299, 0.587, 0.114])

def changeColorImage(image, d):
	sizeH, sizeV, dim = image.shape
	new_image = np.zeros([sizeH, sizeV, dim],dtype=np.uint8)	
	delta = np.array( dim*[d] )
	for i in range(sizeH):
		for j in range(sizeV):
			for k in range(dim):
				v = image[i][j][k] + d
				v = max(0,v)
				v = min(255,v)
				new_image[i][j][k] = v 

	return new_image

name = 'lena.png'

image = cv2.imread(name)

a = 64

im = changeColorImage(image, a)
cv2.imwrite( "lena2.png", im )

im = changeColorImage(image, -a)
cv2.imwrite( "lena3.png", im )

cv2.destroyAllWindows()