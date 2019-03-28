import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import time

def createWhiteDisk2(height=100, width=100, xc=50, yc=50, rc=20):
	xx, yy = np.meshgrid(range(height), range(width))
	img = np.array(((xx - xc) ** 2 + (yy - yc) ** 2 - rc ** 2) < 0).astype('float64')
	return img

def createSineImage2(height, width, freq, theta):
	img = np.zeros((height, width), dtype=np.float64)
	xx, yy = np.meshgrid(range(height), range(width))
	theta = np.deg2rad(theta)
	rho = (xx * np.cos(theta) - yy * np.sin(theta))
	img[:] = np.sin(2 * np.pi * freq * rho)
	cv2.normalize(img, img, 1, 0, cv2.NORM_MINMAX)
	return img

def createCosineImage(height, width, freq, theta):
	img = np.zeros((height, width), dtype=np.float64)
	for x in range(img.shape[0]):
		for y in range(img.shape[1]):
			img[x][y] = np.cos(2 * np.pi * freq * (x * np.cos(theta) - y * np.sin(theta)))
	cv2.normalize(img, img, 1, 0, cv2.NORM_MINMAX)
	return img

img = createSineImage2(100,100,0.1,45)
img2 = createSineImage2(100,100,0.1,-45)
plt.subplot('131')
plt.title('Image 1')
plt.imshow(img, cmap = 'gray')
plt.subplot('132')
plt.title('Image 2')
plt.imshow(img2, cmap = 'gray')
plt.subplot('133')
plt.title('Image 2')
plt.imshow(img2, cmap = 'gray')
plt.show()

##################

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import time

def plotRGB(bgr):
	plt.subplot('221')
	plt.title('B')
	plt.imshow(bgr[0], cmap = 'gray')
	plt.subplot('222')
	plt.title('G')
	plt.imshow(bgr[1], cmap = 'gray')
	plt.subplot('223')
	plt.title('R')
	plt.imshow(bgr[2], cmap = 'gray')
	plt.subplot('224')
	plt.title('Image')
	plt.imshow( cv2.cvtColor(img, cv2.COLOR_BGR2RGB) , cmap = 'gray')
	plt.show()

def plotRGBMCY(bgr):
	r = [1,0,0]
	g = [0,1,0]
	b = [0,0,1]
	c = [0,1,1]
	m = [1,0,1]
	y = [1,1,0]

	img1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	img_g = np.uint8(g * img)
	img_r = np.uint8(r * img)
	img_b = np.uint8(b * img)
	img_c = np.uint8(c * img)
	img_m = np.uint8(m * img)
	img_y = np.uint8(y * img)

	plt.subplot('241')
	plt.title('RGB')
	plt.imshow( img1 )
	plt.subplot('242')
	plt.title('G')
	plt.imshow( img_g )
	plt.subplot('243')
	plt.title('R')
	plt.imshow( img_r )
	plt.subplot('244')
	plt.title('B')
	plt.imshow( img_b )

	plt.subplot('245')
	plt.title('C')
	plt.imshow( img_c )
	plt.subplot('246')
	plt.title('M')
	plt.imshow( img_m )
	plt.subplot('247')
	plt.title('Y')
	plt.imshow( img_y )
	plt.show()

def plotColorMap(img, colormap = 1):
	bgr = cv2.split(img)
	plt.subplot('221')
	plt.title('B')
	plt.imshow( cv2.applyColorMap(bgr[0], colormap))
	plt.subplot('222')
	plt.title('G')
	plt.imshow( cv2.applyColorMap(bgr[1], colormap))
	plt.subplot('223')
	plt.title('R')
	plt.imshow( cv2.applyColorMap(bgr[2], colormap))
	plt.subplot('224')
	plt.title('Image')
	plt.imshow( cv2.cvtColor(img, cv2.COLOR_BGR2RGB) )
	plt.show()

def plotColorMapInv(img, colormap = 1):
	img1 = 255 - img
	bgr = cv2.split(img1)
	plt.subplot('221')
	plt.title('B')
	plt.imshow( cv2.applyColorMap(bgr[0], colormap))
	plt.subplot('222')
	plt.title('G')
	plt.imshow( cv2.applyColorMap(bgr[1], colormap))
	plt.subplot('223')
	plt.title('R')
	plt.imshow( cv2.applyColorMap(bgr[2], colormap))
	plt.subplot('224')
	plt.title('Image')
	plt.imshow( cv2.cvtColor(img1, cv2.COLOR_BGR2RGB) )
	plt.show()

def createWhiteDisk2(height=100, width=100, xc=50, yc=50, rc=20):
	xx, yy = np.meshgrid(range(height), range(width))
	img = np.array(((xx - xc) ** 2 + (yy - yc) ** 2 - rc ** 2) < 0).astype('float64')
	return img


img1 = []

void = createWhiteDisk2(100, 100, 0, 0, 0)

disk1 = createWhiteDisk2(xc=40,yc=60)
img1 = np.zeros((3,100, 100), dtype=np.float64)
img1[0] = disk1

plt.imshow(img1)
plt.show()
disk2 = createWhiteDisk2()
plt.imshow(disk2,cmap = 'gray')
plt.show()
disk3 = createWhiteDisk2(xc=60,yc=60)
plt.imshow(disk3,cmap = 'gray')
plt.show()
