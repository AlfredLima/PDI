import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import time

def doNothing():
	pass

# Funções de leitura de imagem

def readColorImage(path, show=False):
	img = cv2.imread(path, cv2.IMREAD_COLOR)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	if show :	plotImage(img)
	return img

# Mudar o formato da imagem

def bgr2hsv(img):
	return cv2.cvtColor(img, cv2.COLOR_BGR2HSV_FULL)

def hsv2bgr(img):
	return cv2.cvtColor(img, cv2.COLOR_HSV2BGR_FULL)

# Separar canais

def splitChanels(img):
	return cv2.split(img)

# Juntar canais

def mergeChanels(img1, img2, img3):
	shape = img1.shape
	img = np.zeros((shape[0], shape[1], 3), dtype=np.uint8)

	for i in range(shape[0]):
		for j in range(shape[1]):
			img[i][j][0] = img1[i][j]
			img[i][j][1] = img2[i][j]
			img[i][j][2] = img3[i][j]

	return img

# Saturation

def sat(img, value, delta):
	return np.where( abs(img-value) > delta  , 255 , 0 )

# Criação de imagem

def createImgTom(height=100, width=100, value = 0):
	img = np.zeros((height, width), dtype=np.uint8)
	img = np.where( 1 , value , img )
	return img

def createWhiteDisk(height=100, width=100, xc=50, yc=50, rc=20):
	xx, yy = np.meshgrid(range(height), range(width))
	img = np.array(((xx - xc) ** 2 + (yy - yc) ** 2 - rc ** 2) < 0).astype('float64')
	return img

def createBlackDisk(height=100, width=100, xc=50, yc=50, rc=20):
	xx, yy = np.meshgrid(range(height), range(width))
	img = 1 - np.array(((xx - xc) ** 2 + (yy - yc) ** 2 - rc ** 2) < 0).astype('float64')
	return img

def createWhiteRing(height=100, width=100, xc=50, yc=50, re=20, ri=10):
	img1 = createBlackDisk(height,width,xc,yc,re)
	img2 = createBlackDisk(height,width,xc,yc,ri)
	return cv2.bitwise_xor(img2, img1)

def createBlackRing(height=100, width=100, xc=50, yc=50, re=20, ri=10):
	return 1 - createWhiteRing(height,width,xc,yc,re,ri)

def createSineImage(height, width, freq, theta):
	img = np.zeros((height, width), dtype=np.float64)
	xx, yy = np.meshgrid(range(height), range(width))
	theta = np.deg2rad(theta)
	rho = (xx * np.cos(theta) - yy * np.sin(theta))
	img[:] = np.sin(2 * np.pi * freq * rho)
	cv2.normalize(img, img, 1, 0, cv2.NORM_MINMAX)
	return img

def create2DGaussian(height = 100, width = 100, mx = 50, my = 50, sx = 10, sy = 100, theta = 0):
	xx0, yy0 = np.meshgrid(range(height), range(width))
	xx0 -= mx
	yy0 -= my
	theta = np.deg2rad(theta)
	xx = xx0 * np.cos(theta) - yy0 * np.sin(theta)
	yy = xx0 * np.sin(theta) + yy0 * np.cos(theta)
	try:
		img = np.exp( -(xx**2)/(2*sx**2) + (yy**2)/(2*sy**2) )
	except Exception as e:
		img = np.zeros((height,width), dtype='float64')
	cv2.normalize(img,img,1,0, cv2.NORM_MINMAX)
	return img

# Operações

def sumImage(img1, img2):
	img = img1 + img2
	img = np.where( img > 1. , 1. , img )
	img = np.where( img < 0. , 0. , img )
	return img

def dotImage(img1, img2):
	img1_ = np.copy(img1)
	img2_ = np.copy(img2)
	cv2.normalize(img1_,img1_,1,0, cv2.NORM_MINMAX)
	cv2.normalize(img2_,img2_,1,0, cv2.NORM_MINMAX)
	
	shape = img1.shape

	img = np.zeros(shape, dtype=np.float64)
	
	for i in range(shape[0]):
		for j in range(shape[1]):
			img[i][j] = img1_[i][j] * img2_[i][j]

	return img

# FFT

def FFTinImage(img):
	f = np.fft.fft2(img)
	fshift = np.fft.fftshift(f)
	return f, fshift

def magnitude(img):
	return 20*np.log(np.abs(img))

# Histograma

def compute_histogram_1C(src):
	b_hist = cv2.calcHist([src], [0], None, [256], [0, 256], True, False)
	hist_w = 256
	hist_h = 400
	bin_w = np.round(hist_w / 256)
	histImage = np.ones((hist_h, hist_w), np.uint8)
	cv2.normalize(b_hist, b_hist, 0, histImage.shape[0], cv2.NORM_MINMAX)
	
	for i in range(1, 256):
		cv2.line(histImage, (int(bin_w * (i - 1)), int(hist_h - np.round(b_hist[i - 1]))),
		(int(bin_w * i), int(hist_h - np.round(b_hist[i]))), 255, 2, cv2.LINE_8, 0)

	return histImage

# Plots

def plotImage(img, title = 'Image', cmap = 'gray'):
	plt.imshow(img, cmap = cmap )
	plt.title(title)
	plt.show()

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

def plotImageHSV(hsv, title = ''):
	chanel = ['HSV' + title , 'H', 'S', 'V']
	split = splitChanels(hsv)
	imgs = [cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR_FULL)]
	for e in split :
		imgs.append(e)

	for i in range(len(imgs)):
		plt.subplot('22' + str(i+1) )
		plt.title(chanel[i])
		plt.imshow(imgs[i], cmap = 'gray')

	plt.show()

def scaleImage2_uchar(src):
    tmp = np.copy(src)
    if src.dtype != np.float32:
        #tmp = np.float32(tmp)
        pass
    cv2.normalize(tmp, tmp, 1, 0, cv2.NORM_MINMAX)
    tmp = 255 * tmp
    tmp = np.uint8(tmp)
    return tmp
