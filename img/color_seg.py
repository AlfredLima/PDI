#%%
import cv2
import numpy as np
import matplotlib.pyplot as plt

#%%
def doNothing(param):
    pass

def scaleImage2_uchar(src):
    tmp = np.copy(src)
    if src.dtype != np.float32:
        tmp = np.float32(tmp)
    cv2.normalize(tmp, tmp, 1, 0, cv2.NORM_MINMAX)
    tmp = 255 * tmp
    tmp = np.uint8(tmp)
    return tmp

def histogram(src):
    hist = cv2.calcHist([src],[0],None,[256],[0,256])
    return hist

#%% Converting between color spaces - BGR to HSV - Color Segmentation
img = cv2.imread('chips.png', cv2.COLOR_BGR2HSV)
img2 = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
img= cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
hsv = cv2.split(img2)

redmask1 = (160, 200, 200)
redmask2 = (210, 255, 255)
redmask = cv2.inRange(img2, redmask1, redmask2)

bluemask1 = (90, 200, 200)
bluemask2 = (130, 255, 255)
bluemask = cv2.inRange(img2, bluemask1, bluemask2)

yellowmask1 = (15, 200, 200)
yellowmask2 = (30, 255, 255)
yellowmask = cv2.inRange(img2, yellowmask1, yellowmask2)

orangemask1 = (2, 200, 200)
orangemask2 = (10, 255, 255)
orangemask = cv2.inRange(img2, orangemask1, orangemask2)

greenmask1 = (50, 200, 120)
greenmask2 = (80, 255, 200)
greenmask = cv2.inRange(img2, greenmask1, greenmask2)

chipsmask = redmask + bluemask + yellowmask + orangemask + greenmask

resultred = cv2.bitwise_and(img, img, mask=redmask)
resultblue = cv2.bitwise_and(img, img, mask=bluemask)
resultyellow = cv2.bitwise_and(img, img, mask=yellowmask)
resultorange = cv2.bitwise_and(img, img, mask=orangemask)
resultgreen = cv2.bitwise_and(img, img, mask=greenmask)
resultchips = cv2.bitwise_and(img, img, mask=chipsmask)

chips = cv2.cvtColor(resultchips, cv2.COLOR_RGB2HSV)
hsvchips = cv2.split(chips)

print(histogram(hsvchips[0]))

plt.figure()
plt.subplot(231), plt.imshow(img)
plt.subplot(232), plt.imshow(img2)
plt.subplot(233), plt.imshow(hsv[0])
plt.subplot(234), plt.imshow(hsv[1])
plt.subplot(235), plt.imshow(scaleImage2_uchar(hsv[2]))
plt.subplot(236), plt.imshow(resultchips)

plt.figure()
plt.subplot(331), plt.imshow(resultred)
plt.subplot(332), plt.imshow(resultblue)
plt.subplot(333), plt.imshow(resultyellow)
plt.subplot(334), plt.imshow(resultorange)
plt.subplot(335), plt.imshow(resultgreen)
plt.subplot(336), plt.imshow(chips)
plt.subplot(337), plt.imshow(hsvchips[0])
plt.subplot(338), plt.imshow(hsvchips[1])
plt.subplot(339), plt.imshow(scaleImage2_uchar(hsvchips[2]))

plt.figure()
plt.subplot(231), plt.plot(histogram(hsv[0]))
plt.subplot(232), plt.plot(histogram(hsv[1]))
plt.subplot(233), plt.plot(histogram(scaleImage2_uchar(hsv[2])))
plt.subplot(234), plt.plot(histogram(hsvchips[0]))
plt.subplot(235), plt.plot(histogram(hsvchips[1]))
plt.subplot(236), plt.plot(histogram(scaleImage2_uchar(hsvchips[2])))

plt.xlim([0,256])

plt.show()

#%% Verduras
img = cv2.imread('img/5.png', cv2.COLOR_BGR2HSV)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

lower_hue = 0
lower_saturation = 0
lower_value = 0
upper_hue = 255
upper_saturation = 255
upper_value = 255

cv2.namedWindow("img", cv2.WINDOW_KEEPRATIO)
cv2.namedWindow("img2", cv2.WINDOW_KEEPRATIO)
cv2.namedWindow("img3", cv2.WINDOW_KEEPRATIO)

cv2.createTrackbar("lh", "img2", lower_hue, 255, doNothing)
cv2.createTrackbar("uh", "img2", upper_hue, 255, doNothing)
cv2.createTrackbar("ls", "img2", lower_saturation, 255, doNothing)
cv2.createTrackbar("us", "img2", upper_saturation, 255, doNothing)
cv2.createTrackbar("lv", "img2", lower_value, 255, doNothing)
cv2.createTrackbar("uv", "img2", upper_value, 255, doNothing)

while cv2.waitKey(1) != ord('q'):
    lower_hue = cv2.getTrackbarPos("lh", "img2")
    lower_saturation = cv2.getTrackbarPos("ls", "img2")
    lower_value = cv2.getTrackbarPos("lv", "img2")
    upper_hue = cv2.getTrackbarPos("uh", "img2")
    upper_saturation = cv2.getTrackbarPos("us", "img2")
    upper_value = cv2.getTrackbarPos("uv", "img2")

    mask1 = (lower_hue , lower_saturation, lower_value)
    print(mask1)
    mask2 = (upper_hue, upper_saturation, upper_value)
    mask = cv2.inRange(hsv, mask1, mask2)

    img2 = cv2.bitwise_and(img, img, mask=mask)
    cv2.imshow("img", img)
    cv2.imshow("img2", img2)
    cv2.imshow("img3", img-img2)
cv2.destroyAllWindows()

#%% Rosas
img = cv2.imread('img/flowers.jpg', cv2.COLOR_BGR2HSV)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

lower_hue = 0
lower_saturation = 0
lower_value = 0
upper_hue = 255
upper_saturation = 255
upper_value = 255

cv2.namedWindow("img", cv2.WINDOW_KEEPRATIO)
cv2.namedWindow("img2", cv2.WINDOW_KEEPRATIO)
cv2.namedWindow("img3", cv2.WINDOW_KEEPRATIO)

cv2.createTrackbar("lh", "img2", lower_hue, 255, doNothing)
cv2.createTrackbar("uh", "img2", upper_hue, 255, doNothing)
cv2.createTrackbar("ls", "img2", lower_saturation, 255, doNothing)
cv2.createTrackbar("us", "img2", upper_saturation, 255, doNothing)
cv2.createTrackbar("lv", "img2", lower_value, 255, doNothing)
cv2.createTrackbar("uv", "img2", upper_value, 255, doNothing)

while cv2.waitKey(1) != ord('q'):
    lower_hue = cv2.getTrackbarPos("lh", "img2")
    lower_saturation = cv2.getTrackbarPos("ls", "img2")
    lower_value = cv2.getTrackbarPos("lv", "img2")
    upper_hue = cv2.getTrackbarPos("uh", "img2")
    upper_saturation = cv2.getTrackbarPos("us", "img2")
    upper_value = cv2.getTrackbarPos("uv", "img2")

    mask1 = (lower_hue , lower_saturation, lower_value)
    print(mask1)
    mask2 = (upper_hue, upper_saturation, upper_value)
    mask = cv2.inRange(hsv, mask1, mask2)

    img2 = cv2.bitwise_and(img, img, mask=mask)
    cv2.imshow("img", img)
    cv2.imshow("img2", img2)
    cv2.imshow("img3", img-img2)
cv2.destroyAllWindows()

#%%
baboon = cv2.imread('img/baboon.png')
baboonRGB = cv2.cvtColor(baboon, cv2.COLOR_BGR2RGB)
rgb = cv2.split(baboonRGB)
baboonCMY = 255 - baboonRGB
cmy = cv2.split(baboonCMY)

titles = ['R','G','B', 'C', 'M', 'Y']
images = [rgb[0], rgb[1], rgb[2], cmy[0], cmy[1], cmy[2]]
for i in range(6):
    plt.subplot(2,3,i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.show()