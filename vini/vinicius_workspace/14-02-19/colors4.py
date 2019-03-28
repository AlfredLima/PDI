
#%%
import cv2
import numpy as np



def createWhiteDisk(height = 100, width = 100, xc = 50, yc = 50, rc = 20):
    disk = np.zeros((height, width), np.float64)
    for x in range(disk.shape[0]):
        for y in range(disk.shape[1]):
            if (x - xc)*(x - xc)+(y - yc)*(y - yc) <= rc * rc:
                disk[x][y] = 1.0
    return disk
    
def scaleImage2_uchar(src):
    tmp = np.copy(src)
    if src.dtype != np.float32:
        tmp = np.float32(tmp)
    cv2.normalize(tmp, tmp, 1, 0, cv2.NORM_MINMAX)
    tmp = 255 * tmp
    tmp = np.uint8(tmp)
    return tmp


rows = 1e3
radius = rows/4
bx = rows/2
by = rows/2 - radius/2
gx = rows/2 - radius/2
gy = rows/2 + radius/2
rx = rows/2 + radius/2
ry = rows/2 + radius/2

bgr = [
        createWhiteDisk(int(rows), int(rows), int(bx), int(by), int(radius)),
        createWhiteDisk(int(rows), int(rows), int(gx), int(gy), int(radius)),
        createWhiteDisk(int(rows), int(rows), int(rx), int(ry), int(radius))
        ]
cv2.namedWindow('img', cv2.WINDOW_KEEPRATIO)

while 0xFF & cv2.waitKey(1) != ord('q'):
    img = cv2.merge(bgr)
    img = scaleImage2_uchar(img)
    cv2.imshow('img', img)
cv2.destroyAllWindows()




