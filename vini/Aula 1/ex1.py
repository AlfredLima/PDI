import cv2
import numpy as np

def print_image(gray_scale):
    new_image = np.full((3, 400, 400), gray_scale, dtype=np.uint8)

    new_image_red, new_image_green, new_image_blue = new_image

    new_rgb = np.dstack([new_image_red, new_image_green, new_image_blue])

    return new_rgb



cv2.imshow("Black Image", print_image(0))
cv2.imshow("Gray Image", print_image(128))
cv2.waitKey(0)
