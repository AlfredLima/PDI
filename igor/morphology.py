# %% Importações
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import convex_hull_image, skeletonize, thin, reconstruction, disk

# %% Operadores lógicos
img1 = cv2.imread('img/Fig0903(a)(utk).tif', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('img/Fig0903(b)(gt).tif', cv2.IMREAD_GRAYSCALE)

intersection = img1 & img2
union = img1 | img2
complement = ~img1
difference = img1 & ~img2

plt.subplot(231), plt.imshow(img1, cmap='gray')
plt.subplot(232), plt.imshow(img2, cmap='gray')
plt.subplot(233), plt.imshow(complement, cmap='gray')
plt.subplot(234), plt.imshow(union, cmap='gray')
plt.subplot(235), plt.imshow(intersection, cmap='gray')
plt.subplot(236), plt.imshow(difference, cmap='gray')
plt.show()

# %% Erosão - A (-) B: Erosão de A por B
img = cv2.imread("img/Fig0908(a)(wirebond-mask).tif", cv2.IMREAD_GRAYSCALE)
kernel1 = np.ones((11, 11), dtype='uint8')
kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))  # Mesma coisa de fazer manualmente
kernel2 = np.ones((15, 15), dtype='uint8')
kernel3 = np.ones((45, 45), dtype='uint8')

img1 = cv2.erode(img, kernel1, iterations=1)
img2 = cv2.erode(img, kernel2, iterations=1)
img3 = cv2.erode(img, kernel3, iterations=1)

plt.subplot(221), plt.imshow(img, cmap='gray')
plt.subplot(222), plt.imshow(img1, cmap='gray')
plt.subplot(223), plt.imshow(img2, cmap='gray')
plt.subplot(224), plt.imshow(img3, cmap='gray')
plt.show()

# %% Dilatação - A (+) B: Dilatação de A por B
img = cv2.imread('img/Fig0906(a)(broken-text).tif', cv2.IMREAD_GRAYSCALE)
kernel = np.zeros((3, 3), dtype='uint8')
kernel[:, 1] = 1
kernel[1, :] = 1
kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))  # Mesma coisa de fazer manualmente

img1 = cv2.dilate(img, kernel, iterations=1)

plt.subplot(121), plt.imshow(img, cmap='gray')
plt.subplot(122), plt.imshow(img1, cmap='gray')
plt.show()

# %% Filtragem de ruído usando abertura e fechamento
# Abertura - A () B: Abertura de A por B = Erosão de A por B seguida de Dilatação B
# Suaviza contornos e elimina coisas estreitas
# Fechamento - A (.) B: Fechamento de A por B = Dilatação de A por B seguida de Erosão por B
# Suaviza contornos e une coisas estreitas

img = cv2.imread('img/Fig0911(a)(noisy-fingerprint).tif', cv2.IMREAD_GRAYSCALE)
kernel = np.ones((3, 3), dtype='uint8')

# Erosão
img1 = cv2.erode(img, kernel, iterations=1)
# Abertura
img2 = cv2.dilate(img1, kernel, iterations=1)
img2 = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)  # Mesma coisa de fazer manualmente
# Dilatação da abertura
img3 = cv2.dilate(img2, kernel, iterations=1)
# Erosão da dilatação da abertura = Fechamento da abertura
img4 = cv2.erode(img3, kernel, iterations=1)
img4 = cv2.morphologyEx(img2, cv2.MORPH_CLOSE, kernel)  # Mesma coisa de fazer manualmente

plt.subplot(231), plt.imshow(img, cmap='gray')
plt.subplot(232), plt.imshow(img1, cmap='gray')
plt.subplot(233), plt.imshow(img2, cmap='gray')
plt.subplot(234), plt.imshow(img3, cmap='gray')
plt.subplot(235), plt.imshow(img4, cmap='gray')
plt.show()

# %% Transformação Hit or Miss - Ferramenta para identificação de formato = A (*) B
# Código para detecção de corners
input_image = np.array(([0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 255, 255, 255, 0, 0, 0, 255],
                        [0, 255, 255, 255, 0, 0, 0, 0],
                        [0, 255, 255, 255, 0, 255, 0, 0],
                        [0, 0, 255, 0, 0, 0, 0, 0],
                        [0, 0, 255, 0, 0, 255, 255, 0],
                        [0, 255, 0, 255, 0, 0, 255, 0],
                        [0, 255, 255, 255, 0, 0, 0, 0]), dtype="uint8")

k1 = np.array(([0, 1, 0],
               [-1, 1, 1],
               [-1, -1, 0]), dtype="int")
k2 = np.array(([-1, -1, 0],
               [-1, 1, 1],
               [0, 1, 0]), dtype="int")
k3 = np.array(([0, 1, 0],
               [1, 1, -1],
               [0, -1, -1]), dtype="int")
k4 = np.array(([0, -1, -1],
               [1, 1, -1],
               [0, 1, 0]), dtype="int")

output_image1 = cv2.morphologyEx(input_image, cv2.MORPH_HITMISS, k1)
output_image2 = cv2.morphologyEx(input_image, cv2.MORPH_HITMISS, k2)
output_image3 = cv2.morphologyEx(input_image, cv2.MORPH_HITMISS, k3)
output_image4 = cv2.morphologyEx(input_image, cv2.MORPH_HITMISS, k4)
output_image = output_image1 | output_image2 | output_image3 | output_image4

plt.subplot(121), plt.imshow(input_image, cmap='gray'), plt.title('I')
plt.subplot(122), plt.imshow(output_image, cmap='gray'), plt.title('O')
plt.show()

# %% Extração de borda AKA Gradiente Morfológico
# TopHat = Input - Opening
# BlackHat = Input - Closing
img = cv2.imread('img/lincoln.tif', cv2.IMREAD_GRAYSCALE)
b = np.ones((3, 3), np.uint8)
dilated = cv2.morphologyEx(img, cv2.MORPH_DILATE, b)
eroded = cv2.morphologyEx(img, cv2.MORPH_ERODE, b)
d = dilated & ~img
e = img & ~eroded

morphological_gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, b)

plt.subplot(231), plt.imshow(img, cmap='gray'), plt.title('img')
plt.subplot(232), plt.imshow(dilated, cmap='gray'), plt.title('Dilated')
plt.subplot(233), plt.imshow(eroded, cmap='gray'), plt.title('Eroded')
plt.subplot(234), plt.imshow(d, cmap='gray'), plt.title('Dilated - IMG')
plt.subplot(235), plt.imshow(e, cmap='gray'), plt.title('IMG - Eroded')
plt.subplot(236), plt.imshow(morphological_gradient, cmap='gray'), plt.title('Morph_Grad')
plt.show()

# %% Preenchimento de buracos
img = cv2.imread('img/region-filling-reflections.tif', cv2.IMREAD_GRAYSCALE)
mask = np.zeros((img.shape[0] + 2, img.shape[1] + 2), np.uint8)

col = [0, 160, 180, 300]
row = [0, 250, 200, 240]

result = img.copy()
idx = 0
cv2.floodFill(result, mask, (row[idx], col[idx]), 255)
result_inv = cv2.bitwise_not(result)
saida = img | result_inv

plt.subplot(231), plt.imshow(img, cmap='gray'), plt.title('A')
plt.subplot(232), plt.imshow(~img, cmap='gray'), plt.title('$A^C$')
plt.subplot(233), plt.imshow(result, cmap='gray'), plt.title('$R = (X_{k-1} \oplus B)$')
plt.subplot(234), plt.imshow(result_inv, cmap='gray'), plt.title('$R^C$')
plt.subplot(235), plt.imshow(saida, cmap='gray')
plt.show()

# %% Detectando componentes conectados e printando a quantidade de pixels de cada um deles
img = cv2.imread('img/chickenfillet.tif', cv2.IMREAD_GRAYSCALE)
ret, thresh = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
thresh_eroded = cv2.erode(thresh, kernel, iterations=1)

output = cv2.connectedComponents(thresh_eroded)
num = output[1].max()

for i in range(1, num + 1):
    pts = np.where(output[1] == i)
    print(i, len(pts[1]))
    if len(pts[0]) < 50:  # Removendo elementos conectados pequenos
        output[1][pts] = 0

plt.subplot(211), plt.imshow(img, cmap='gray')
plt.subplot(212), plt.imshow(thresh_eroded, cmap='gray')

# %% Convex Hull - Casco convexo
img = cv2.imread('img/horse2.png', cv2.IMREAD_GRAYSCALE)
chull = convex_hull_image(img)
plt.subplot(121), plt.imshow(img, cmap='gray'), plt.title('Original')
plt.subplot(122), plt.imshow(chull, cmap='gray'), plt.title('ConvexHull')

# %% Convex Hull - Mão aberta e mão fechada
open_hand = cv2.imread('img/hand1.jpg', cv2.IMREAD_GRAYSCALE)
closed_hand = cv2.imread('img/hand2.jpg', cv2.IMREAD_GRAYSCALE)
open_hand_hull = convex_hull_image(open_hand).astype(np.uint8)
closed_hand_hull = convex_hull_image(closed_hand).astype(np.uint8)

plt.subplot(221), plt.imshow(open_hand, cmap='gray'), plt.title('Open Hand')
plt.subplot(222), plt.imshow(closed_hand, cmap='gray'), plt.title('Closed Hand')
plt.subplot(223), plt.imshow(open_hand_hull, cmap='gray'), plt.title('Open Hand Hull')
plt.subplot(224), plt.imshow(closed_hand_hull, cmap='gray'), plt.title('Closed Hand Hull')

count1 = 0
count2 = 0

for i in open_hand_hull:
    for j in i:
        if j:
            count1 += 1

for i in closed_hand_hull:
    for j in i:
        if j:
            count2 += 1

if count1 > count2:
    print("A primeira mão está aberta")
else:
    print("A segunda mão está aberta")

# %% Tamanho do objeto usando contornos
open_hand = cv2.imread('img/hand1.jpg', cv2.IMREAD_GRAYSCALE)
closed_hand = cv2.imread('img/hand2.jpg', cv2.IMREAD_GRAYSCALE)
ret, thresh1 = cv2.threshold(open_hand, 127, 255, 0)
ret, thresh2 = cv2.threshold(closed_hand, 127, 255, 0)
contours1, hierarchy = cv2.findContours(thresh1, 1, 2)
contours2, hierarchy = cv2.findContours(thresh2, 1, 2)
cnt1 = contours1[0]
cnt2 = contours2[0]
area1 = cv2.contourArea(cnt1)
area2 = cv2.contourArea(cnt2)
perimeter = cv2.arcLength(cnt1, True)  # contorno aberto ou fechado
# aproximação de contorno
epsilon = 0.1 * cv2.arcLength(cnt1, True)
approx = cv2.approxPolyDP(cnt1, epsilon, True)
# hull na mao
hull = cv2.convexHull(cnt1)
# contorno convexo
k = cv2.isContourConvex(cnt1)

# %% convex hull open hand

img_open = cv2.imread("img/hand1.png", cv2.IMREAD_GRAYSCALE)
blur = cv2.blur(img_open, (3, 3))  # blur the image
ret, thresh = cv2.threshold(blur, 50, 255, cv2.THRESH_BINARY)
plt.subplot(131), plt.imshow(img_open, cmap='gray'), plt.title('Original')
# plt.subplot(132), plt.imshow(blur, cmap='gray'), plt.title('Blur')
# plt.subplot(133), plt.imshow(thresh, cmap='gray'), plt.title('Thresh')

# %% convex hull continuation open hand

contours_open, hierarchy_open = cv2.findContours(img_open.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# create hull array for convex hull points
hull_open = []

# calculate points for each contour
for i in range(len(contours_open)):
    # creating convex hull object for each contour
    hull_open.append(cv2.convexHull(contours_open[i], False))

# create an empty black image
img_open_hull = np.zeros((img_open.shape[0], img_open.shape[1], 3), np.uint8)
open_hull_img = np.zeros((img_open.shape[0], img_open.shape[1], 3), np.uint8)

# draw contours and hull points
for i in range(len(contours_open)):
    color_contours = (0, 255, 0)  # green - color for contours
    color = (255, 0, 0)  # blue - color for convex hull
    # draw ith contour
    cv2.drawContours(img_open_hull, contours_open, i, color_contours, 1, 8, hierarchy_open)
    # draw ith convex hull object
    cv2.drawContours(img_open_hull, hull_open, i, color, 1, 8)

    cv2.drawContours(open_hull_img, hull_open, i, color, 1, 8)

# %% plotting convex hull open hand
plt.subplot(131), plt.imshow(img_open, cmap='gray'), plt.title('Original')
plt.subplot(132), plt.imshow(open_hull_img, cmap='gray'), plt.title('Convex hull')
plt.subplot(133), plt.imshow(img_open_hull, cmap='gray'), plt.title('Convex hull and hand')

# %% convex hull close hand

img_close = cv2.imread("img/hand2.jpg", cv2.IMREAD_GRAYSCALE)
# blur = cv2.blur(img_close, (3, 3)) # blur the image
# ret, thresh = cv2.threshold(blur, 50, 255, cv2.THRESH_BINARY)
plt.subplot(131), plt.imshow(img_close, cmap='gray'), plt.title('Original')
# plt.subplot(132), plt.imshow(blur, cmap='gray'), plt.title('Blur')
# plt.subplot(133), plt.imshow(thresh, cmap='gray'), plt.title('Thresh')

# %% convex hull continuation close hand

contours_close, hierarchy_close = cv2.findContours(img_close, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# create hull array for convex hull points
hull_close = []

# calculate points for each contour
for i in range(len(contours_close)):
    # creating convex hull object for each contour
    hull_close.append(cv2.convexHull(contours_close[i], False))

# create an empty black image
img_close_hull = np.zeros((img_close.shape[0], img_close.shape[1], 3), np.uint8)
close_hull_img = np.zeros((img_close.shape[0], img_close.shape[1], 3), np.uint8)

# draw contours and hull points
for i in range(len(contours_close)):
    color_contours = (0, 255, 0)  # green - color for contours
    color = (255, 0, 0)  # blue - color for convex hull
    # draw ith contour
    cv2.drawContours(img_close_hull, contours_close, i, color_contours, 1, 8, hierarchy_close)
    # draw ith convex hull object
    cv2.drawContours(img_close_hull, hull_close, i, color, 1, 8)

    cv2.drawContours(close_hull_img, hull_close, i, color, 1, 8)

# %% plotting convex hull close hand
plt.subplot(131), plt.imshow(img_close, cmap='gray'), plt.title('Original')
plt.subplot(132), plt.imshow(close_hull_img, cmap='gray'), plt.title('Convex hull')
plt.subplot(133), plt.imshow(img_close_hull, cmap='gray'), plt.title('Convex hull and hand')

# %% Esqueleto na mãozinea
img_orig = cv2.imread('img/horse2.png', 0)

ret, img = cv2.threshold(img_orig, 127, 255, 0)

size = np.size(img)
skel = np.zeros(img.shape, np.uint8)

element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
done = False

while (not done):
    eroded = cv2.erode(img, element)
    temp = cv2.dilate(eroded, element)
    temp = cv2.subtract(img, temp)
    skel = cv2.bitwise_or(skel, temp)
    img = eroded.copy()

    zeros = size - cv2.countNonZero(img)
    if zeros == size:
        done = True

plt.subplot(121), plt.imshow(img_orig, cmap='gray'), plt.title('Original')
plt.subplot(122), plt.imshow(skel, cmap='gray'), plt.title('Skeleton')

# %% Esqueletinho no easy
img = cv2.imread('img/horse2.png', cv2.IMREAD_GRAYSCALE)
img[img == 255] = 1
skeleton = skeletonize(img)
thinned = thin(img)
thinned_partial = thin(img, max_iter=25)
plt.subplot(221), plt.imshow(img, cmap='gray'), plt.title('Original')
plt.subplot(222), plt.imshow(skeleton, cmap='gray'), plt.title('Skeleton')
# Thinning = A (X) B = A - (A (*) B) - Imagem - hitmiss
# Thickening = A ((.)) B = Imagem união Hitmiss
plt.subplot(223), plt.imshow(thinned, cmap='gray'), plt.title('Thin')
plt.subplot(224), plt.imshow(thinned_partial, cmap='gray'), plt.title('Partial Thin')

# %% Convex Hull
hand = cv2.imread("img/hand1.jpg", 0)

ret, threshold = cv2.threshold(hand, 10, 255, cv2.THRESH_BINARY)

contours, hiearchy = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
basic_contours = cv2.drawContours(hand, contours, -1, (0, 255, 0))

hull = [cv2.convexHull(c) for c in contours]
final = cv2.drawContours(hand, hull, -1, (255, 255, 255))

plt.subplot(221), plt.imshow(hand, cmap='gray'), plt.title('Original')
plt.subplot(222), plt.imshow(threshold, cmap='gray'), plt.title('Threshold')
plt.subplot(223), plt.imshow(final, cmap='gray'), plt.title('Convex Hull')

# %% Reconstrução morfológica
img = cv2.imread('img/Fig0930(a)(calculator).tif', cv2.IMREAD_GRAYSCALE)
kernel = np.ones((1, 71), np.uint8)
kernel1 = np.ones((1, 11), np.uint8)
kernel2 = np.ones((1, 21), np.uint8)

img_erosion = cv2.morphologyEx(img, cv2.MORPH_ERODE, kernel)  # marker, img = mask
img_opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

img_opening_by_reconstruction = reconstruction(img_erosion, img)
img_top_hat_by_reconstruction = img - img_opening_by_reconstruction
img_top_hat = img - img_opening
img_top_hat_by_reconstruction_erosion = cv2.morphologyEx(img_top_hat_by_reconstruction, cv2.MORPH_ERODE, kernel1)
img_top_hat_opening_by_reconstruction = reconstruction(img_top_hat_by_reconstruction_erosion, img_top_hat)
img_top_hat_opening_by_reconstruction_dilatation = cv2.morphologyEx(img_top_hat_opening_by_reconstruction,
                                                                    cv2.MORPH_DILATE, kernel2)
minimum = np.minimum(img_top_hat_by_reconstruction, img_top_hat_opening_by_reconstruction_dilatation)
final = reconstruction(minimum, img_top_hat_opening_by_reconstruction_dilatation)

plt.subplot(331), plt.imshow(img, cmap='gray'), plt.title('Original')
plt.subplot(332), plt.imshow(img_opening_by_reconstruction, cmap='gray'), plt.title('Opening by reconstruction')
plt.subplot(333), plt.imshow(img_opening, cmap='gray'), plt.title('Opening')
plt.subplot(334), plt.imshow(img_top_hat_by_reconstruction, cmap='gray'), plt.title('Top-Hat by reconstruction')
plt.subplot(335), plt.imshow(img_top_hat, cmap='gray'), plt.title('Top-Hat')
plt.subplot(336), plt.imshow(img_top_hat_opening_by_reconstruction, cmap='gray'), plt.title(
    'Top-Hat opening by reconstruction')
plt.subplot(337), plt.imshow(img_top_hat_opening_by_reconstruction_dilatation, cmap='gray'), plt.title(
    'Top-Hat opening by reconstruction dilatation')
plt.subplot(338), plt.imshow(minimum, cmap='gray'), plt.title('Minimum d and g')
plt.subplot(339), plt.imshow(final, cmap='gray'), plt.title('Final')
plt.show()

# %% Reconstrução morfológica
img = cv2.imread('img/Fig0922(a)(book-text).tif', cv2.IMREAD_GRAYSCALE)
kernel = np.ones((51, 1), np.uint8)

eroded_img = cv2.erode(img, kernel, iterations=1)
opening_img = cv2.dilate(eroded_img, kernel, iterations=1)
final_img = reconstruction(eroded_img, img)
plt.subplot(221), plt.imshow(img, cmap='gray'), plt.title('Original')
plt.subplot(222), plt.imshow(eroded_img, cmap='gray'), plt.title('Erosion')
plt.subplot(223), plt.imshow(opening_img, cmap='gray'), plt.title('Opening')
plt.subplot(224), plt.imshow(final_img, cmap='gray'), plt.title('Opening by Reconstruction')

# %% Correção de sombreamento usando top-hat
img = cv2.imread('img/Fig0926(a)(rice).tif', cv2.IMREAD_GRAYSCALE)
ret, img_threshold = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
img_opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
img_tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
ret, tophat_threshold = cv2.threshold(img_tophat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
plt.subplot(231), plt.imshow(img, cmap='gray'), plt.title('Original')
plt.subplot(232), plt.imshow(img_threshold, cmap='gray'), plt.title('Original Threshold')
plt.subplot(233), plt.imshow(img_opening, cmap='gray'), plt.title('Opening')
plt.subplot(234), plt.imshow(img_tophat, cmap='gray'), plt.title('TopHat')
plt.subplot(235), plt.imshow(tophat_threshold, cmap='gray'), plt.title('TopHat - Thresholded')

# %% Image thresholding
img = cv2.imread('img/lena.png', 0)
ret, thresh1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
ret, thresh2 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
ret, thresh3 = cv2.threshold(img, 127, 255, cv2.THRESH_TRUNC)
ret, thresh4 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO)
ret, thresh5 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO_INV)
# th2 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_MEAN_C,\
#            cv.THRESH_BINARY,11,2)
# th3 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,\
#            cv.THRESH_BINARY,11,2)
titles = ['Original Image', 'BINARY', 'BINARY_INV', 'TRUNC', 'TOZERO', 'TOZERO_INV']
images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]
for i in range(6):
    plt.subplot(2, 3, i + 1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
plt.show()

# global thresholding
# ret1,th1 = cv.threshold(img,127,255,cv.THRESH_BINARY)
# Otsu's thresholding
# ret2,th2 = cv.threshold(img,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
# Otsu's thresholding after Gaussian filtering
# blur = cv.GaussianBlur(img,(5,5),0)
# ret3,th3 = cv.threshold(blur,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)

# %% Granulometria
img = cv2.imread('img/Fig0925(a)(dowels).tif', cv2.IMREAD_GRAYSCALE)
kernel = disk(5)
kernel1 = disk(10)
kernel2 = disk(20)
kernel3 = disk(25)
kernel4 = disk(30)
img_smoothed = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
img_smoothed = cv2.morphologyEx(img_smoothed, cv2.MORPH_CLOSE, kernel)

op10 = cv2.morphologyEx(img_smoothed, cv2.MORPH_OPEN, kernel1)
op20 = cv2.morphologyEx(img_smoothed, cv2.MORPH_OPEN, kernel2)
op25 = cv2.morphologyEx(img_smoothed, cv2.MORPH_OPEN, kernel3)
op30 = cv2.morphologyEx(img_smoothed, cv2.MORPH_OPEN, kernel4)

titles = ['Original Image', 'Smoothed', 'Opening 10', 'Opening 20', 'Opening 25', 'Opening 30']
images = [img, img_smoothed, op10, op20, op25, op30]
for i in range(6):
    plt.subplot(2, 3, i + 1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
plt.show()

# %% Segmentação por textura
img = cv2.imread('img/2.png', cv2.IMREAD_GRAYSCALE)
thresholded = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 5, 0)
eroded = cv2.morphologyEx(thresholded, cv2.MORPH_ERODE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
                          iterations=4)
only_balls = reconstruction(eroded, thresholded)

kernel = disk(25)
eroded_balls = cv2.morphologyEx(only_balls, cv2.MORPH_ERODE, kernel, iterations=1)
big_balls = reconstruction(eroded_balls, only_balls)
small_balls = only_balls - big_balls

titles = ['Original Image', 'Balls', 'Big Balls', 'Small Balls']
images = [img, only_balls, big_balls, small_balls]
for i in range(4):
    plt.subplot(2, 2, i + 1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
plt.show()

# %% Separação galáxia
img = cv2.imread('img/1.png', cv2.IMREAD_GRAYSCALE)
eroded = cv2.morphologyEx(img, cv2.MORPH_ERODE, disk(1), iterations=1)
final = img - eroded

titles = ['Original Image', 'Eroded', 'Final']
images = [img, eroded, final]
for i in range(3):
    plt.subplot(2, 2, i + 1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
plt.show()

# remove_small_holes , param
# remove small objects , param