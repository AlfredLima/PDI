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

#pid.plotRGBMCY(img)

r = [1, 0, 0]
g = [0, 1, 0]
b = [0, 0, 1]
c = [0, 1, 1]
m = [1, 0, 1]
y = [1, 1, 0]

img_g = np.uint8(g * img)
img_r = np.uint8(r * img)
img_b = np.uint8(b * img)
img_c = np.uint8(c * img)
img_m = np.uint8(m * img)
img_y = np.uint8(y * img)

img2_g = np.uint8(g * img2)
img2_r = np.uint8(r * img2)
img2_b = np.uint8(b * img2)
img2_c = np.uint8(c * img2)
img2_m = np.uint8(m * img2)
img2_y = np.uint8(y * img2)


c1, c2, c3 = pid.splitChanels(img)

colors = [c1, c2, c3]

for i in colors:
    for j in colors:
        for k in colors:
            img3 = pid.mergeChanels(i, j, k)
            imgs = [pid.bgr2rgb(img), pid.bgr2rgb(img2), pid.bgr2rgb(img3)]
            pid.showImages(imgs)