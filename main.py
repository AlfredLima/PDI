import Library

pid = Library.Library()

# Logic Operation - Test

logic = False

if logic:
    img1 = pid.readGrayScaleImage('img/utk.tif', True)
    img2 = pid.readGrayScaleImage('img/gt.tif', True)

    img3 = pid.notImage(img1)
    img4 = pid.orImage(img1, img2)
    img5 = pid.andImage(img1, img2)
    img6 = pid.diffImage(img1, img2)

    pid.showImage(img3)
    pid.showImage(img4)
    pid.showImage(img5)
    pid.showImage(img6)

# Operation Morphological - Test

morphological = False

if morphological:
    img1 = pid.readGrayScaleImage('img/text.tif', True)
    img2 = pid.readGrayScaleImage('img/wirebond.tif', True)
    size = (3, 3)
    kernel1 = pid.createGrayScaleImage(size, scale=1)
    kernel1[1][:] = 0
    kernel1[0][1] = 0
    kernel1[2][1] = 0

    img3 = pid.dilateImage(img1, kernel1)
    pid.showImage(img3)

    # sizes = (11,11) , (15,15) and (45,45)
    size = (11, 11)
    kernel2 = pid.createGrayScaleImage(size, scale=1)
    img4 = pid.erosionImage(img2, kernel2)
    pid.showImage(img4)

    img5 = pid.hitOrMissImage(img1, kernel1, kernel2)
    pid.showImage(img5)

    img6 = pid.readGrayScaleImage('img/lincoln.tif', True)
    size = (3, 3)
    kernel3 = pid.createGrayScaleImage(size, scale=0, show=True)
    img7 = pid.boundaryExtractionImage(img6, kernel3)
    pid.showImage(img7)

    img8 = pid.readGrayScaleImage('img/region-filling-reflections.tif', show=True)
    size = (25, 25)
    kernel4 = pid.createGrayScaleImage(size, scale=1)
    img9 = pid.holeFillingImage(img8, kernel4, kernel4, 1, 2)
    pid.showImage(img9)

    img10 = pid.readGrayScaleImage('img/noisy-fingerprint.tif', show=True)
    size = (3, 3)
    kernel5 = pid.createGrayScaleImage(size=size, scale=1)
    img11 = pid.removingNoise(img10, kernel5, kernel5, 1, 1)
    pid.showImage(img11)

    img10 = pid.readGrayScaleImage('img/noisy-fingerprint.tif', show=True)
    size = (3, 3)
    kernel5 = pid.createGrayScaleImage(size=size, scale=1)
    img11 = pid.removingNoise(img10, kernel5, kernel5)
    pid.showImage(img11)

#

# Convex Hull

convexHull = False
if convexHull:

    img12 = pid.readGrayScaleImage('img/horse2.png')
    size = (15, 15)
    img13, img14 = pid.convexHullSimple(img12)
    img15 = pid.convexHullFull(img12)
    imgs = [img12, img13, img14, img15]
    pid.showImages(imgs)

# Skeleton

skeleton = False
if skeleton:
    img16 = pid.readGrayScaleImage('img/horse2.png')
    img17 = pid.skeletonImage(img16)
    img18 = pid.skeletonImage(img16)

    imgs = [img16, img17, img18]
    pid.showImages(imgs)

pid.close()