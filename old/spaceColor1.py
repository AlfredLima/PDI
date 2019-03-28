### 
# import cv2
# flags = [i for i in dir(cv2) if i.startswith('COLOR_')]
# print (flags)


####
import cv2
import numpy as np

def nothing(x):
    pass


cap = cv2.VideoCapture(0)

while(1):

    # Take each frame
    _, frame = cap.read()

    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # # define range of color in HSV

    lower_red1 = np.array([0,124,124])
    upper_red1 = np.array([10,255,255])
    lower_red2 = np.array([170,124,124])
    upper_red2 = np.array([180,255,255])

    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)

    lower_green = np.array([60,100,100])
    upper_green = np.array([90,255,255])

    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    lower_blue = np.array([90,100,100])
    upper_blue = np.array([150,255,255])

    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

    mask = mask_red1 | mask_red2 | mask_green | mask_blue

    res = cv2.bitwise_and(frame,frame, mask= mask)

    cv2.imshow('frame',frame)
    cv2.imshow('res',res)
    
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()

