import numpy as np
import cv2

def getSubImage(image, sizeH, sizeV, dx = 0, dy = 0):
	new_image = np.zeros([sizeH,sizeV,3],dtype=np.uint8)
	sizeh, sizev, _ = image.shape
	for x in range( dx, min(sizeH+dx, sizeh) ):
		for y in range( dy, min(sizeV+dy, sizev) ):
			new_image[x-dx][y-dy] = image[x][y]
	return new_image

name = 'baboon.png'

image = cv2.imread(name)

im = getSubImage(image, 256, 256, dx=256, dy=128)
cv2.imwrite( "baboon2.png", im )

cv2.destroyAllWindows()