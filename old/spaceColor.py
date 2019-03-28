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

cv2.namedWindow('res', cv2.WINDOW_KEEPRATIO)

cv2.createTrackbar('H_min', 'res', 0, 255, nothing)
cv2.createTrackbar('H_max', 'res', 255, 255, nothing)
cv2.createTrackbar('S_min', 'res', 0, 255, nothing)
cv2.createTrackbar('S_max', 'res', 255, 255, nothing)
cv2.createTrackbar('V_min', 'res', 0, 255, nothing)
cv2.createTrackbar('V_max', 'res', 255, 255, nothing)

while(1):

    # Take each frame
    _, frame = cap.read()

    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # blue = np.uint8([[[255,0,0]]])
    # green = np.uint8([[[0,255,0]]])
    # red = np.uint8([[[0,0,255]]])

    # hsv_green = cv2.cvtColor(green,cv2.COLOR_BGR2HSV)
    # hsv_blue = cv2.cvtColor(blue,cv2.COLOR_BGR2HSV)
    # hsv_red = cv2.cvtColor(red,cv2.COLOR_BGR2HSV)

    # print(hsv_blue, hsv_green, hsv_red)

    # # define range of blue color in HSV
    # lower_blue = np.array([90,150,150])
    # upper_blue = np.array([150,255,255])

    # lower_green = np.array([40,150,150])
    # upper_green = np.array([90,255,255])

    # lower_red = np.array([0,150,150])
    # upper_red = np.array([40,255,255])


    # # Threshold the HSV image to get only blue colors
    # mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    # mask_green = cv2.inRange(hsv, lower_green, upper_green)
    # mask_red = cv2.inRange(hsv, lower_red, upper_red)

    # # Bitwise-AND mask and original image

    h_min = cv2.getTrackbarPos('H_min', 'res')
    h_max = cv2.getTrackbarPos('H_max', 'res')
    s_min = cv2.getTrackbarPos('S_min', 'res')
    s_max = cv2.getTrackbarPos('S_max', 'res')
    v_min = cv2.getTrackbarPos('V_min', 'res')
    v_max = cv2.getTrackbarPos('V_max', 'res')
    
    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])
    
    mask = cv2.inRange(hsv, lower, upper)

    res = cv2.bitwise_and(frame,frame, mask= mask)

    # cv2.imshow('frame',frame)
    # cv2.imshow('mask',mask)
    cv2.imshow('res',res)
    
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()

