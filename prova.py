import Library
import numpy as np

pid = Library.Library()
'''
# Questão 1
img = pid.readColorImage('img/flowers.jpg', show=True)
hsv = pid.bgr2hsv(img)
col = pid.get_color_img(hsv, img, 'yellow')
col_rgb = pid.bgr2rgb(col)
pid.plotImage(col_rgb)

# Questão 2
img = pid.readColorImage('img/baboon.png', show=True)
pid.plotRGBMCY(img)

'''
# Questão 3
img = pid.readColorImage('img/baboon.png', show=True)
img2 = pid.readColorImage('img/abc.png', show=True)

img_rgb = pid.bgr2rgb(img)
img2_rgb = pid.bgr2rgb(img2)

#%%
import cv2
import numpy as np

#%%
baboonRGB = cv2.imread("img/baboon.png", cv2.COLOR_BGR2RGB)
baboonABC = cv2.imread("img/abc.png")

r1 = baboonRGB[0][0][0]
b1 = baboonRGB[0][0][1]
g1 = baboonRGB[0][0][2]

r2 = baboonRGB[0][1][0]
g2 = baboonRGB[0][1][1]
b2 = baboonRGB[0][1][2]

r3 = baboonRGB[0][2][0]
g3 = baboonRGB[0][2][1]
b3 = baboonRGB[0][2][2]

a1 = baboonABC[0][0][0]
ab1 = baboonABC[0][0][1]
c1 = baboonABC[0][0][2]

a2 = baboonABC[0][1][0]
ab2 = baboonABC[0][1][1]
c2 = baboonABC[0][1][2]

a3 = baboonABC[0][2][0]
ab3 = baboonABC[0][2][1]
c3 = baboonABC[0][2][2]

l = np.array([[r1, g1, b1], [r2, g2, b2], [r3, g3, b3]])
s1 = np.array([a1, a2, a3])
s2 = np.array([ab1, ab2, ab3])
s3 = np.array([c1, c2, c3])

x1 = np.linalg.solve(l, s1)
x2 = np.linalg.solve(l, s2)
x3 = np.linalg.solve(l, s3)

M = np.matrix([x1, x2, x3])

abc = np.zeros((512, 512, 3), dtype=np.float64)

for i in range(512):
    for j in range(512):
        abc[i][j] = np.dot(M, baboonRGB[i][j])/255

cv2.imshow("img", abc)
cv2.waitKey(0)