import cv2
import numpy as np

im = cv2.imread("lena.png")

brightness_value = 100
darkness_value = 100
im_lght = np.where((255 - im) < brightness_value,255,im+brightness_value)
im_dkn = np.where(im < darkness_value, 0, im - darkness_value)

cv2.imshow("Mais escura", im_dkn)
cv2.imshow("Normal", im)
cv2.imshow("Mais clara", im_lght)
cv2.waitKey(0)