import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import time

def doNothing():
	pass

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

def compute_histogram_1C(src):
    # Compute the histograms:
    b_hist = cv2.calcHist([src], [0], None, [256], [0, 256], True, False)

    # Draw the histograms for B, G and R
    hist_w = 512
    hist_h = 400
    bin_w = np.round(hist_w / 256)

    histImage = np.ones((hist_h, hist_w), np.uint8)

    # Normalize the result to [ 0, histImage.rows ]
    cv2.normalize(b_hist, b_hist, 0, histImage.shape[0], cv2.NORM_MINMAX)

    # Draw for each channel
    for i in range(1, 256):
        cv2.line(histImage, (int(bin_w * (i - 1)), int(hist_h - np.round(b_hist[i - 1]))),
                 (int(bin_w * i), int(hist_h - np.round(b_hist[i]))), 255, 2, cv2.LINE_8, 0)

    return histImage


img = cv2.imread('img/chips.png', cv2.IMREAD_COLOR)
img2 = cv2.cvtColor(img, cv2.COLOR_BGR2HSV_FULL)
hsv = cv2.split(img2)

'''
for i in range(3):
	cv2.normalize(hsv[i], hsv[i], 1, 0, cv2.NORM_MINMAX)

plt.imshow(hsv[0], cmap = 'gray')
plt.show()
plt.imshow(hsv[1], cmap = 'gray')
plt.show()
plt.imshow(hsv[2], cmap = 'gray')
plt.show()
plt.imshow(compute_histogram_1C(hsv[0]), cmap = 'gray')
plt.show()

'''

'''
img = cv2.imread('img/baboon.png')

xsize = 3
ysize = 3

cv2.namedWindow('img', cv2.WINDOW_KEEPRATIO)
cv2.namedWindow('img2', cv2.WINDOW_KEEPRATIO)

cv2.createTrackbar('xsize', 'img2', xsize, 50, doNothing)
cv2.createTrackbar('ysize', 'img2', ysize, 50, doNothing)


while cv2.waitKey(1) != ord('q'):
	xsize = cv2.getTrackbarPos('xsize', 'img2')
	ysize = cv2.getTrackbarPos('ysize', 'img2')

	img2 = cv2.blur(img, (xsize+1,ysize+1))
	cv2.imshow('img', img)
	cv2.imshow('img2', img2)

	pass
'''

img = cv2.imread('img/baboon.png')

wsize = 3

cv2.namedWindow('img', cv2.WINDOW_KEEPRATIO)
cv2.namedWindow('img2', cv2.WINDOW_KEEPRATIO)

cv2.createTrackbar('wsize', 'img2', wsize, 10, doNothing)

while cv2.waitKey(1) != ord('q'):
	wsize = cv2.getTrackbarPos('wsize', 'img2')
	
	img2 = cv2.Laplacian(img, cv2.CV_16S, ksize=2*wsize+1, scale=1,delta=0, borderType=cv2.BORDER_DEFAULT )
	cv2.imshow('img', img)
	cv2.imshow('img2', img2)

	pass


cv2.destroyAllWindows()
