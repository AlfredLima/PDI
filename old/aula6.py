import cv2
import matplotlib.pyplot as plt
import numpy as np

def doNothing():
	pass

def createWhiteDisk2(height = 100, width = 100, xc = 50, yc = 50, rc = 20):
        xx, yy = np.meshgrid( range(height), range(width) )
        img = np.array( (xx-xc)**2 + (yy-yc)**2 - rc**2 < 0 ).astype('float64')
        return img
        
     
def createCosineImage2(height = 100, width = 100, freq = 0, theta = 0):
    img = np.zeros((height,width), dtype=np.float64)
    xx, yy = np.meshgrid( range(height), range(width) )
    theta = np.deg2rad(theta)
    rho = xx * np.cos(theta) - yy * np.sin(theta)
    img[:] = np.cos( 2 * np.pi * freq * rho )
    return img

def scaleImage2_uchar(src):
	tmp = np.copy(src)
	if src.dtype != np.float32:
		tmp = np.float32(tmp)
	cv2.normalize(tmp, tmp, 1, 0, cv2.NORM_MINMAX)
	tmp = 255*tmp
	tmp = np.uint8(tmp)
	return tmp

def create2DGaussian(rows = 100, cols = 100, mx = 50, my = 50, sx = 10, sy = 100, theta = 0):
	xx0, yy0 = np.meshgrid(range(rows), range(cols))
	xx0 -= mx
	yy0 -= my
	theta = np.deg2rad(theta)
	xx = xx0 * np.cos(theta) - yy0 * np.sin(theta)
	yy = xx0 * np.sin(theta) + yy0 * np.cos(theta)
	try:
		img = np.exp( -(xx**2)/(2*sx**2) + (yy**2)/(2*sy**2) )
	except Exception as e:
		img = np.zeros((rows,cols), dtype='float64')
	cv2.normalize(img,img,1,0, cv2.NORM_MINMAX)
	return img

#%%%

img = createWhiteDisk2(400,400)     
cv2.imshow("img", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

#%%%

img = createCosineImage2(400,400, 10, 30)     
cv2.imshow("img", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

#%%%

img = create2DGaussian()     
cv2.imshow("img", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

rows = 100
cols = 100
tetha = 0
xc = 50
yc = 50
sx = 30
sy = 10
theta = 0

cv2.namedWindow('img', cv2.WINDOW_KEEPRATIO)
cv2.createTrackbar('xc', 'img', xc, int(rows), doNothing)
cv2.createTrackbar('yc', 'img', yc, int(cols), doNothing)
cv2.createTrackbar('sx', 'img', sx, int(rows), doNothing)
cv2.createTrackbar('sy', 'img', sx, int(cols), doNothing)

while 0xFF & cv2.waitKey(1) != ord('q'):
	xc = cv2.getTrackbarPos('xc', 'img')
	yc = cv2.getTrackbarPos('yc', 'img')
	sx = cv2.getTrackbarPos('sx', 'img')
	sy = cv2.getTrackbarPos('sy', 'img')
	theta = create2DGaussian(rows, cols, xc, yc, sx, sy, theta)
	cv2.imshow('img', cv2.applyColorMap(scaleImage2_uchar(img),cv2.COLORMAP_JET) )
	pass

cv2.destroyAllWindows()

img = cv2.imread('rectangle.jpg',0)

planes = [np.zeros(img.shape, dtype=np.float64), np.zeros(img.shape, dtype=np.float64)]
planes[0][:] = np.float64(img[:])

img2 = cv2.merge(planes)
img2 = cv2.dft(img2)

planes = cv2.split(img2)

cv2.normalize(planes[0],planes[0], 1,0, cv2.NORM_MINMAX)
cv2.normalize(planes[1],planes[1], 1,0, cv2.NORM_MINMAX)

while  0xFF & cv2.waitKey(1) != ord('q'):
	cv2.imshow('Original', img)
	cv2.imshow('Plane 0 - Real', planes[0])
	cv2.imshow('Plane 1 - Imaginary', planes[1])

cv2.destroyAllWindows()
