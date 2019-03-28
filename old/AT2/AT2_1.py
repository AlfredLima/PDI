import numpy as np
import cv2

SIZE = 400
BLACK = 0
GRAY = 128

black_image = np.zeros([SIZE,SIZE,3],dtype=np.uint8)
black_image.fill(BLACK)

gray_image = np.zeros([SIZE,SIZE,3],dtype=np.uint8)
gray_image.fill(GRAY)

cv2.imwrite( "BLACK.jpg", black_image )

cv2.imwrite( "GRAY.jpg", gray_image )

cv2.destroyAllWindows()