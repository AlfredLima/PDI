import cv2
import matplotlib.pyplot as plt
import numpy as np
import skimage.morphology as sk

class Library():

    def __init__(self):
        pass

    def doNothing(self, x):
        pass

    def showImage(self, img, name="img"):
        cv2.imshow(name, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def showVideo(self, img, name="Video"):
        cv2.imshow(name, img)

    def waitKey(self, time):
        return cv2.waitKey(time)

    def destroyAllWindows(self):
        cv2.destroyAllWindows()

    def readGrayScaleImage(self, path, show=False):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if show:    self.showImage(img)
        return img

    def readColorImage(self, path, show=False):
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if show:    self.showImage(img)
        return img

    def readEachChannel(self, path, show=False):
        img = self.readColorImage(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        rgb = cv2.split(img)
        if show:
            plt.subplot("141")
            plt.title("Original")
            plt.imshow(img)
            plt.subplot("142")
            plt.title("R")
            plt.imshow(rgb[0])
            plt.subplot("143")
            plt.title("G")
            plt.imshow(rgb[1])
            plt.subplot("144")
            plt.title("B")
            plt.imshow(rgb[2])
            plt.show()
        return rgb

    def histogramGrayScale(self, img, show=False):
        if show:
            plt.subplot("211");
            plt.title("Original");
            plt.imshow(img, 'gray')
            plt.subplot("212");
            plt.title("Histogram");
            plt.hist(img.ravel(), 256, [0, 256])
            plt.show()

    def histogramColor(self, img, show=False):
        color = ('b', 'g', 'r')
        for i, col in enumerate(color):
            histr = cv2.calcHist([img], [i], None, [256], [0, 256])
            plt.plot(histr, color=col)
            plt.xlim([0, 256])
        plt.show()

    def createGrayScaleImage(self, size, scale=0, show=False):
        img = np.ones(size, dtype=np.float64) * scale
        if show:
            cv2.imshow("img", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return img

    def createGrayScaleRandomImage(self, size, show=False):
        img = np.ones(size, dtype=np.float64)
        cv2.randu(img, 0, 255)
        if show:
            cv2.imshow("img", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return img

    def createColorRandomImage(self, size, show=False):
        new_size = (size[0], size[1], 3)
        img = np.ones(new_size, dtype=np.uint8)
        bgr = cv2.split(img)
        for i in range(3):
            cv2.randu(bgr[i], 0, 255)
        img = cv2.merge(bgr)
        if show:
            cv2.imshow("img", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return img

    def createGrayScaleRandomNormallyImage(self, size, show=False):
        img = np.ones(size, dtype=np.uint8)
        cv2.randn(img, 127, 40)
        histr = cv2.calcHist([img], [], None, [255], [0, 255])
        if show:
            plt.plot(histr)
            plt.xlim([0, 256])
            plt.show()
        return img

    def createColorRandomNormallyImage(self, size, show=False):
        new_size = (size[0], size[1], 3)
        img = np.ones(new_size, dtype=np.uint8)
        bgr = cv2.split(img)
        cv2.randn(bgr[0], 127, 40)
        cv2.randn(bgr[1], 127, 40)
        cv2.randn(bgr[2], 127, 40)
        img = cv2.merge(bgr)
        if show:
            color = ('b', 'g', 'r')
            for i, col in enumerate(color):
                histr = cv2.calcHist([img], [i], None, [256], [0, 256])
                plt.plot(histr, color=col)
                plt.xlim([0, 256])
            plt.show()
        return img

    def addScalarImage(self, img, scalar=0, show=False):
        img2 = np.asarray(img, dtype=np.float64)
        img2 = img2 + scalar
        img2 = np.where(img2 > 255, 255, img2)
        img2 = np.asarray(img2, dtype=np.uint8)
        if show:
            plt.subplot("121");
            plt.title("IMG 1");
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.subplot("122");
            plt.title("IMG 2");
            plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
            plt.show()
        return img2

    def sumImages(self, img1, img2, scalar1=1, scalar2=1, show=False):
        img1 = np.asarray(img1, dtype=np.float64)
        img2 = np.asarray(img2, dtype=np.float64)
        img3 = scalar1 * img1 + scalar2 * img2
        img3 = np.where(img3 > 255, 255, img3)

        img1 = np.asarray(img1, dtype=np.uint8)
        img2 = np.asarray(img2, dtype=np.uint8)
        img3 = np.asarray(img3, dtype=np.uint8)

        if show:
            plt.subplot("131")
            plt.title("IMG 1")
            plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
            plt.subplot("132")
            plt.title("IMG 2")
            plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
            plt.subplot("133")
            plt.title("IMG 3")
            plt.imshow(cv2.cvtColor(img3, cv2.COLOR_BGR2RGB))
            plt.show()

        return img3

    def maxImages(self, img1, img2, show=False):
        img3 = cv2.max(img1, img2)

        if show:
            plt.subplot("131")
            plt.title("IMG 1")
            plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
            plt.subplot("132")
            plt.title("IMG 2")
            plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
            plt.subplot("133")
            plt.title("IMG 3")
            plt.imshow(cv2.cvtColor(img3, cv2.COLOR_BGR2RGB))
            plt.show()

        return img3

    def absdiffImages(self, img1, img2, show=False):
        img3 = cv2.absdiff(img1, img2)

        if show:
            plt.subplot("131")
            plt.title("IMG 1")
            plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
            plt.subplot("132")
            plt.title("IMG 2")
            plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
            plt.subplot("133")
            plt.title("IMG 3")
            plt.imshow(cv2.cvtColor(img3, cv2.COLOR_BGR2RGB))
            plt.show()

        return img3

    def addNoiseinImage(self, img, show=False):
        noise = self.createColorRandomImage(img.shape)
        img2 = self.sumImages(img, noise)

        if show:
            plt.subplot("121")
            plt.title("IMG 1")
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.subplot("122")
            plt.title("IMG 2")
            plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
            plt.show()

        return img2

    def addSaltPepperinImage(self, img, show=False):
        size = (img.shape[0], img.shape[1])
        noise = self.createGrayScaleRandomImage(size)
        salt = noise > 250
        pepper = noise < 5
        img2 = img.copy()
        img2[salt == True] = 255
        img2[pepper == True] = 0

        if show:
            plt.subplot("121")
            plt.title("IMG 1")
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.subplot("122")
            plt.title("IMG 2")
            plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
            plt.show()

        return img2

    def addSaltPepperinImage(self, img, show=False):
        size = (img.shape[0], img.shape[1])
        noise = self.createGrayScaleRandomImage(size)
        salt = noise > 250
        pepper = noise < 5
        img2 = img.copy()
        img2[salt == True] = 255
        img2[pepper == True] = 0

        if show:
            plt.subplot("121")
            plt.title("IMG 1")
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.subplot("122")
            plt.title("IMG 2")
            plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
            plt.show()

        return img2

    def andImages(self, img1, img2, show=False):
        and_img = img1 & img2

        if show:
            plt.subplot("131")
            plt.title("IMG 1")
            plt.imshow(img1, 'gray')
            plt.subplot("132")
            plt.title("IMG 2")
            plt.imshow(img2, 'gray')
            plt.subplot("153")
            plt.title("AND")
            plt.imshow(and_img, 'gray')
            plt.show()

        return and_img

    def orImages(self, img1, img2, show=False):
        or_img = img1 | img2

        if show:
            plt.subplot("131")
            plt.title("IMG 1")
            plt.imshow(img1, 'gray')
            plt.subplot("132")
            plt.title("IMG 2")
            plt.imshow(img2, 'gray')
            plt.subplot("153")
            plt.title("OR")
            plt.imshow(or_img, 'gray')
            plt.show()

        return or_img

    def orImages(self, img1, img2, show=False):
        not_img = ~img

        if show:
            plt.subplot("131")
            plt.title("IMG 1")
            plt.imshow(img1, 'gray')
            plt.subplot("132")
            plt.title("IMG 2")
            plt.imshow(img2, 'gray')
            plt.subplot("153")
            plt.title("NOT")
            plt.imshow(not_img, 'gray')
            plt.show()

        return not_img

    def logTransformImg(self, img, scalar=1, norm=0, show=False):
        img1 = np.asarray(img, dtype=np.float64)
        img1 = img1 / 255
        img2 = np.ones(img.shape, np.float64)
        img2 = scalar * np.log(1 + img1)

        if norm:
            cv2.normalize(img2, img2, 1, 0, cv2.NORM_MINMAX)
        else:
            img2 = np.where(img2 > 1, 1, img2)

        img2 = img2 * 255
        img2 = np.asarray(img2, dtype=np.uint8)

        if show:
            cv2.imshow("img", img)
            cv2.imshow("img2", img2)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return img2

    def gammaTransformImg(self, img, exp=1, show=False):
        img1 = np.asarray(img, dtype=np.float64)
        img1 = img1 / 255
        img1 = img1 ** exp  # np.power(img1,exp)
        img1 = img1 * 255
        img1 = np.asarray(img1, dtype=np.uint8)

        if show:
            cv2.imshow("img", img)
            cv2.imshow("img1", img1)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return img1

    def testGammaTransform(self, img):
        img1 = np.asarray(img, dtype=np.float64)
        img2 = np.ones(img.shape, np.uint8)
        cv2.namedWindow("img2", cv2.WINDOW_KEEPRATIO)
        n = 0
        cv2.createTrackbar("n", "img2", n, 40, self.doNothing)

        while cv2.waitKey(1) != ord('q'):
            n = cv2.getTrackbarPos("n", "img2") / 10
            img2 = self.gammaTransformImg(img1, exp=n)
            cv2.imshow("img", img)
            cv2.imshow("img2", img2)

        cv2.destroyAllWindows()

    def compute_piecewise_linear_val(self, val, r1, s1, r2, s2):
        output = 0
        if (0 <= val) and (val <= r1):
            output = (s1 / r1) * val
        if (r1 <= val) and (val <= r2):
            output = ((s2 - s1) / (r2 - r1)) * (val - r1) + s1
        if (r2 <= val) and (val <= 1):
            output = ((1 - s2) / (1 - r2)) * (val - r2) + s2

        return output

    def compute_histogram_1C(self, src):
        b_hist = cv2.calcHist([src], [0], None, [256], [0, 256], True, False)
        hist_w = 512
        hist_h = 400
        bin_w = np.round(hist_w / 256)
        histImage = np.ones((hist_h, hist_w), np.uint8)
        cv2.normalize(b_hist, b_hist, 0, histImage.shape[0], cv2.NORM_MINMAX)
        for i in range(1, 256):
            cv2.line(histImage, (int(bin_w * (i - 1)), int(hist_h - np.round(b_hist[i - 1]))),
                     (int(bin_w * i), int(hist_h - np.round(b_hist[i]))), 255, 2, cv2.LINE_8, 0)

        return histImage

    def resizeImage(self, img, size):
        new_size = (size[0], size[1])
        return cv2.resize(img, new_size, 0, 0, cv2.INTER_LINEAR)

    def thresholdingImage(self, img, lim=(150, 240), show=False):
        img2 = np.copy(img)
        img2 = np.where((lim[0] < img2) & (img2 < lim[1]), img2, 0)

        if show:
            cv2.imshow("img", img2)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return img2

    def bitSlicingImage(self, img, slices, show=False):
        mask = 0
        for bit in slices:
            mask = mask | (1 << bit)

        img2 = img & mask

        if show:
            cv2.imshow("img", img)
            cv2.waitKey(0)
            cv2.imshow("img", img2)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return img2

    def histogramEqualization(self, img, show=False):
        img2 = cv2.equalizeHist(img)
        hist = self.compute_histogram_1C(img)
        hist2 = self.compute_histogram_1C(img2)

        if show:
            cv2.imshow("img", img)
            cv2.imshow("img2", img2)
            cv2.imshow("hist", hist)
            cv2.imshow("hist2", hist2)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return img2

    def localHistogramEqualization(self, img, size=(1, 1), show=False):
        img2 = np.zeros(img.shape, img.dtype)
        hsize, vsize = size

        for x in range(hsize, img.shape[0] - hsize):
            for y in range(vsize, img.shape[1] - vsize):
                cv2.equalizeHist(img[y - vsize: y + vsize][x - hsize: x + hsize],
                                 img2[y - vsize: y + vsize][x - hsize: x + hsize])

        if show:
            cv2.imshow("img", img)
            cv2.imshow("img2", img2)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return img2

    def andImage(self, img1, img2):
        return img1 & img2

    def orImage(self, img1, img2):
        return img1 | img2

    def notImage(self, img):
        return ~img

    def diffImage(self, img1, img2):
        return self.andImage(img1, self.notImage(img2))

    # Rectangular Kernel cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
    # Elliptical Kernel cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    # Cross-shaped Kernel cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5))

    def erosionImage(self, img, kernel, iterations=1):
        # - ball
        kernel = np.array(kernel, dtype=np.uint8)
        return cv2.erode(img, kernel=kernel, iterations=iterations)

    def dilateImage(self, img, kernel, iterations=1):
        # + ball
        kernel = np.array(kernel, dtype=np.uint8)
        return cv2.dilate(img, kernel=kernel, iterations=iterations)

    def openingImage(self, img, kernel):
        # clean ball -> dilate( erose(A,B) , B )
        kernel = np.array(kernel, dtype=np.uint8)
        return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

    def closeImage(self, img, kernel):
        # black ball -> erose( dilate(A,B) , B )
        kernel = np.array(kernel, dtype=np.uint8)
        return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

    def hitOrMissImage(self, img, kernel1, kernel2):
        # * ball -> erose(A,B1) and erose(~A,B2) | erose(A,B1) - dilate(A,B2_hat)
        kernel1 = np.array(kernel1, dtype=np.uint8)
        kernel2 = np.array(kernel2, dtype=np.uint8)
        img1 = self.erosionImage(img, kernel1)
        img2 = self.erosionImage(self.notImage(img), kernel2)
        return self.andImage(img1, img2)

    def boundaryExtractionImage(self, img, kernel):
        kernel = np.array(kernel, dtype=np.uint8)
        return self.diffImage(img, self.erosionImage(img, kernel))

    def holeFillingImage(self, img, k1, k2, i1, i2):
        k1 = np.array(k1, dtype=np.uint8)
        k2 = np.array(k2, dtype=np.uint8)
        dilated = self.dilateImage(img=img, kernel=k1, iterations=i1)
        return self.erosionImage(img=dilated, kernel=k2, iterations=i2)

    def removingNoise(self, img, k1, k2):
        k1 = np.array(k1, dtype=np.uint8)
        k2 = np.array(k2, dtype=np.uint8)
        openning = self.openingImage(img=img, kernel=k1)
        return self.closeImage(img=openning, kernel=k2)

    def topHatImage(self, img, kernel):
        kernel = np.array(kernel, dtype=np.uint8)
        openning = self.openingImage(img=img, kernel=kernel)
        return self.diffImage(img1=img, img2=openning)

    def blackHatImage(self, img, kernel):
        return cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)

    def gradientImage(self, img, kernel):
        kernel = np.array(kernel, dtype=np.uint8)
        dilate = self.dilateImage(img=img,kernel=kernel)
        erose = self.erosionImage(img=img,kernel=kernel)
        return self.diffImage(dilate, erose)

    def convexHullSimple(self, img):
        hand = img.copy()

        ret, threshold = cv2.threshold(hand, 10, 255, cv2.THRESH_BINARY)

        contours, hiearchy = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        basic_contours = cv2.drawContours(hand, contours, -1, (0, 255, 0))

        hull = [cv2.convexHull(c) for c in contours]
        final = cv2.drawContours(hand, hull, -1, (255, 255, 255))
        poly = np.zeros((threshold.shape[0], threshold.shape[1], 3), np.uint8)
        poly = cv2.drawContours(poly, hull, -1, (255, 255, 255))

        return poly, final

    def convexHullFull(self, img):
        return sk.convex_hull_image(img)

    def showImages(self, imgs):

        length = len(imgs)
        size_x = 1
        size_y = length

        if length > 3:
            size_x = 2
            size_y = size_y/2
            if length%size_x:
                size_y = size_y + 1

        for idx, img in enumerate(imgs):
            subplot = 100*size_x + 10*size_y + idx+1
            plt.subplot(subplot)
            plt.imshow(img, cmap='gray')
            plt.title(str(idx+1))

        plt.show()

    def skeletonImage(self, img_orig):
        ret, img = cv2.threshold(img_orig, 127, 255, 0)
        size = np.size(img)
        skel = np.zeros(img.shape, np.uint8)

        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        done = False

        while not done:
            eroded = cv2.erode(img, element)
            temp = cv2.dilate(eroded, element)
            temp = cv2.subtract(img, temp)
            skel = cv2.bitwise_or(skel, temp)
            img = eroded.copy()

            zeros = size - cv2.countNonZero(img)
            if zeros == size:
                done = True

        return skel

    def skeletonImage2(self, img):
        return sk.skeletonize(img)

    # Mudar o formato da imagem

    def bgr2hsv(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2HSV_FULL)

    def hsv2bgr(self, img):
        return cv2.cvtColor(img, cv2.COLOR_HSV2BGR_FULL)

    def bgr2rgb(self, img):
        return cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    # Separar canais

    def splitChanels(self, img):
        return cv2.split(img)

    # Juntar canais

    def mergeChanels(self, img1, img2, img3):
        shape = img1.shape
        img = np.zeros((shape[0], shape[1], 3), dtype=np.uint8)

        for i in range(shape[0]):
            for j in range(shape[1]):
                img[i][j][0] = img1[i][j]
                img[i][j][1] = img2[i][j]
                img[i][j][2] = img3[i][j]

        return img

    def createWhiteDisk(self, height=100, width=100, xc=50, yc=50, rc=20):
        xx, yy = np.meshgrid(range(height), range(width))
        img = np.array(((xx - xc) ** 2 + (yy - yc) ** 2 - rc ** 2) < 0).astype('float64')
        return img

    def createBlackDisk(self, height=100, width=100, xc=50, yc=50, rc=20):
        xx, yy = np.meshgrid(range(height), range(width))
        img = 1 - np.array(((xx - xc) ** 2 + (yy - yc) ** 2 - rc ** 2) < 0).astype('float64')
        return img

    def createWhiteRing(self, height=100, width=100, xc=50, yc=50, re=20, ri=10):
        img1 = self.createBlackDisk(height, width, xc, yc, re)
        img2 = self.createBlackDisk(height, width, xc, yc, ri)
        return cv2.bitwise_xor(img2, img1)

    def createBlackRing(self, height=100, width=100, xc=50, yc=50, re=20, ri=10):
        return 1 - self.createWhiteRing(height, width, xc, yc, re, ri)

    def createSineImage(self, height, width, freq, theta):
        img = np.zeros((height, width), dtype=np.float64)
        xx, yy = np.meshgrid(range(height), range(width))
        theta = np.deg2rad(theta)
        rho = (xx * np.cos(theta) - yy * np.sin(theta))
        img[:] = np.sin(2 * np.pi * freq * rho)
        cv2.normalize(img, img, 1, 0, cv2.NORM_MINMAX)
        return img

    def plotImage(self, img, title='Image', cmap='gray'):
        plt.imshow(img, cmap=cmap)
        plt.title(title)
        plt.show()

    def plotRGB(self, bgr):
        plt.subplot('221')
        plt.title('B')
        plt.imshow(bgr[0], cmap='gray')
        plt.subplot('222')
        plt.title('G')
        plt.imshow(bgr[1], cmap='gray')
        plt.subplot('223')
        plt.title('R')
        plt.imshow(bgr[2], cmap='gray')
        plt.subplot('224')
        plt.title('Image')
        plt.imshow(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), cmap='gray')
        plt.show()

    def plotImageHSV(self, hsv, title=''):
        chanel = ['HSV' + title, 'H', 'S', 'V']
        split = self.splitChanels(hsv)
        imgs = [cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB_FULL)]
        for e in split:
            imgs.append(e)

        for i in range(len(imgs)):
            plt.subplot('22' + str(i + 1))
            plt.title(chanel[i])
            plt.imshow(imgs[i], cmap='gray')

        plt.show()

    def scaleImage2_uchar(self, src):
        tmp = np.copy(src)
        if src.dtype != np.float32:
            # tmp = np.float32(tmp)
            pass
        cv2.normalize(tmp, tmp, 1, 0, cv2.NORM_MINMAX)
        tmp = 255 * tmp
        tmp = np.uint8(tmp)
        return tmp

    def sat(self, img, value, delta):
        return np.where(abs(img - value) > delta, 255, 0)

    def get_color_img(self, hsv, img, color):
        img2 = img.copy()
        mask = 0
        if color == "red":
            lower = np.array([169, 100, 100])
            upper = np.array([189, 255, 255])
            mask = mask | cv2.inRange(hsv, lower, upper)
            lower = np.array([0, 100, 100])
            upper = np.array([20, 255, 255])
        elif color == "yellow":
            lower = np.array([20, 100, 100])
            upper = np.array([30, 255, 255])
        elif color == "green":
            lower = np.array([35, 60, 60])
            upper = np.array([80, 255, 255])
        elif color == "blue":
            lower = np.array([110, 50, 50])
            upper = np.array([130, 255, 255])
        elif color == "orange":
            lower = np.array([0, 70, 50])
            upper = np.array([10, 255, 255])
        else:
            return None
        mask = mask | cv2.inRange(hsv, lower, upper)
        return cv2.bitwise_and(img2, img2, mask=mask)

    def plotRGBMCY(self, img):
        r = [1, 0, 0]
        g = [0, 1, 0]
        b = [0, 0, 1]
        c = [0, 1, 1]
        m = [1, 0, 1]
        y = [1, 1, 0]

        img1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_g = np.uint8(g * img)
        img_r = np.uint8(r * img)
        img_b = np.uint8(b * img)
        img_c = np.uint8(c * img)
        img_m = np.uint8(m * img)
        img_y = np.uint8(y * img)

        plt.subplot('241')
        plt.title('RGB')
        plt.imshow(img1)
        plt.subplot('242')
        plt.title('G')
        plt.imshow(img_g)
        plt.subplot('243')
        plt.title('R')
        plt.imshow(img_r)
        plt.subplot('244')
        plt.title('B')
        plt.imshow(img_b)

        plt.subplot('245')
        plt.title('C')
        plt.imshow(img_c)
        plt.subplot('246')
        plt.title('M')
        plt.imshow(img_m)
        plt.subplot('247')
        plt.title('Y')
        plt.imshow(img_y)
        plt.show()

    def close(self):
        cv2.destroyAllWindows()  # fecha todas a janelas abertas
