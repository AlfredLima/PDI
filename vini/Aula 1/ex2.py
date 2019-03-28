import cv2

im = cv2.imread("baboon.png")
sub_image = im[233:476, 127:353]

cv2.imshow("Image", im)
cv2.imshow("Sub Image", sub_image)
cv2.waitKey(0)