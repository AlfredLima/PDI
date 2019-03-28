import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import time

folder = 'Aulas_PDI/Provas/Prova 1/2018_1_AB1'
#%% 
#Q1
def doNothing(x):
    pass


def createWhiteDisk(height=100, width=100, xc=50, yc=50, rc=20):
    disk = np.zeros((height, width), np.float64)
    for x in range(disk.shape[0]):
        for y in range(disk.shape[1]):
            if (x - xc) * (x - xc) + (y - yc) * (y - yc) <= rc * rc:
                disk[x][y] = 1.0
    return disk


def createWhiteDisk2(height=100, width=100, xc=50, yc=50, rc=20):
    xx, yy = np.meshgrid(range(height), range(width))
    img = np.array(
        ((xx - xc) ** 2 + (yy - yc) ** 2 - rc ** 2) < 0).astype('float64')
    return img


def scaleImage2_uchar(src):
    tmp = np.copy(src)
    if src.dtype != np.float32:
        tmp = np.float32(tmp)
    cv2.normalize(tmp, tmp, 1, 0, cv2.NORM_MINMAX)
    tmp = 255 * tmp
    tmp = np.uint8(tmp)
    return tmp


def createCosineImage(height, width, freq, theta):
    img = np.zeros((height, width), dtype=np.float64)
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            img[x][y] = np.cos(
                2 * np.pi * freq * (x * np.cos(theta) - y * np.sin(theta)))
    return img


def createSineImage2(height, width, freq, theta):
    img = np.zeros((height, width), dtype=np.float64)
    xx, yy = np.meshgrid(range(height), range(width))
    theta = np.deg2rad(theta)
    rho = (xx * np.cos(theta) - yy * np.sin(theta))
    img[:] = np.sin(2 * np.pi * freq * rho)
    return img


def applyLogTransform(img):
    img2 = np.copy(img)
    img2 += 1
    img2 = np.log(img2)
    return img2


def create2DGaussian(rows=100,
                     cols=100,
                     mx=50,
                     my=50,
                     sx=10,
                     sy=100,
                     theta=0):
    xx0, yy0 = np.meshgrid(range(rows), range(cols))
    xx0 -= mx
    yy0 -= my
    theta = np.deg2rad(theta)
    xx = xx0 * np.cos(theta) - yy0 * np.sin(theta)
    yy = xx0 * np.sin(theta) + yy0 * np.cos(theta)
    try:
        img = np.exp(- ((xx ** 2) / (2 * sx ** 2) +
                        (yy ** 2) / (2 * sy ** 2)))
    except ZeroDivisionError:
        img = np.zeros((rows, cols), dtype='float64')

    cv2.normalize(img, img, 1, 0, cv2.NORM_MINMAX)
    return img
#%% 
#Q2
rows = 100
cols = 100
theta = 45
img = np.zeros((rows, cols), dtype=np.float64)
xx, yy = np.meshgrid(np.linspace(-cols/2, cols/2 - 1, cols),
                     np.linspace(-rows/2, rows/2 - 1, rows))
cv2.namedWindow('img', cv2.WINDOW_KEEPRATIO)
cv2.createTrackbar("theta", "img", theta, 360, doNothing)
while 0xFF & cv2.waitKey(1) != ord('q'):
    theta = cv2.getTrackbarPos("theta", "img")
    theta2 = np.deg2rad(theta) # convert theta from deg (int) to rad (float)
    img[:] = (xx * np.cos(theta2) - yy * np.sin(theta2))
    img = (img - img.min()) / (img.max() - img.min())
    img2 = scaleImage2_uchar(img)
    cv2.imshow('img', img2)
cv2.destroyAllWindows()
# cv2.imwrite('degrade_theta_' + str(theta) + '_deg.png', img2)

plt.imshow(img2, cmap = 'gray')
plt.xlabel('Theta = ' + str(theta) + ' deg')
plt.show()
#%%
rows = 500
cols = 500
n = 30
np.random.seed(np.int64(time.time()))
xi = np.random.randint(0, cols, n)
yi = np.random.randint(0, rows, n)
radii = np.random.randint(10, 30, n)
img = np.zeros((rows, cols), dtype = np.uint8) != 0
for i in range(n):
    disk = createWhiteDisk2(rows, cols, xi[i], yi[i], radii[i]) != 0
    img = img | disk
plt.imshow(img, 'gray')
plt.show()
# cv2.imwrite('random-circles.png', scaleImage2_uchar(img))

#%% 
#Q3
p = 10

bgr0 = cv2.imread(os.path.join(folder, 'baboon.png'), cv2.IMREAD_COLOR)
noise = np.random.rand(bgr0.shape[0], bgr0.shape[1], bgr0.shape[2])
cv2.namedWindow('img', cv2.WINDOW_KEEPRATIO)
cv2.createTrackbar('noiseDensity', 'img', p, 100, doNothing)
while 0xFF & cv2.waitKey(1) != ord('q'):
    p = cv2.getTrackbarPos('noiseDensity', 'img')
    bgr = bgr0.copy()
    bgr[noise > 1 - p/100] = 255
    bgr[noise < p/100] = 0
    bgr_clean = cv2.medianBlur(bgr, 3)
    cv2.imshow('img', bgr)
    cv2.imshow('clean', bgr_clean)
cv2.destroyAllWindows()
cv2.imwrite(os.path.join(folder, 'baboon_' + 'noise.png'), bgr)
cv2.imwrite(os.path.join(folder, 'baboon_' + 'cleaned.png'), bgr_clean)

plt.subplot('121'); plt.imshow(bgr2rgb(bgr))
plt.subplot('122'); plt.imshow(bgr2rgb(bgr_clean))
plt.show()

#%% 
#Q4

rows = 60
cols = 60
theta = 0
xc = 30
yc = 30
sx = 12
sy = 6
theta = 0

img = cv2.imread(os.path.join(folder, 'messi.jpg'), cv2.IMREAD_GRAYSCALE)
img = img.astype(np.float64)

cv2.namedWindow('img', cv2.WINDOW_KEEPRATIO)
cv2.createTrackbar('sx', 'img', sx, int(rows), doNothing)
cv2.createTrackbar('sy', 'img', sy, int(cols), doNothing)
cv2.createTrackbar('theta', 'img', theta, 360, doNothing)

while 0xFF & cv2.waitKey(1) != ord('q'):
    sx = cv2.getTrackbarPos('sx', 'img')
    sy = cv2.getTrackbarPos('sy', 'img')
    theta = cv2.getTrackbarPos('theta', 'img')

    mask = create2DGaussian(rows, cols, xc, yc, sx + 1, sy + 1, theta)
    img2 = scaleImage2_uchar(cv2.filter2D(img, -1, mask, cv2.BORDER_DEFAULT))

    img2[img2.shape[0] - mask.shape[0]  : ,
         : mask.shape[1]] = scaleImage2_uchar(mask)
    cv2.imshow('img', scaleImage2_uchar(img2))
cv2.destroyAllWindows()

#%% 
#Q5
def compFourSerSqWave(height, width, freq, theta, n=1):
    if n < 1:
        n = 1
    img = np.zeros((height, width), np.float32)
    for i in range(n):
        img = img + (1 / (2 * i + 1)) * createSineImage2(rows,
                                                         cols, (2 * i + 1) * freq, theta=0)
    return img


rows = 400
cols = 400
f0 = 1 / 1e2
ncoeff = 0
cv2.namedWindow('img2', cv2.WINDOW_KEEPRATIO)
cv2.createTrackbar("ncoeff", "img2", ncoeff, 19, doNothing)
while 0xFF & cv2.waitKey(1) != ord('q'):
    ncoeff = cv2.getTrackbarPos('ncoeff', 'img2')
    img2 = compFourSerSqWave(rows, cols, f0, 0, ncoeff)
    cv2.imshow('img2', cv2.applyColorMap(scaleImage2_uchar(img2),
                                         cv2.COLORMAP_OCEAN))
cv2.destroyAllWindows()
plt.imshow(img2, cmap='jet')
plt.title('Number of coefficients: ' + str(ncoeff))
plt.show()

for i in range(7):
    plt.subplot('23' + str(i))
    plt.title(str(i))
    plt.imshow(compFourSerSqWave(rows, cols, f0, 0, i))
plt.show()