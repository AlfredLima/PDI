import cv2
import matplotlib.pyplot as plt
import numpy as np

class PDI():

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

	def	waitKey(self, time):
		return cv2.waitKey(time)

	def destroyAllWindows(self):
		cv2.destroyAllWindows()

	def readGrayScaleImage(self, path, show=False):
		img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
		if show :	self.showImage(img)
		return img

	def readColorImage(self, path, show=False):
		img = cv2.imread(path, cv2.IMREAD_COLOR)
		if show :	self.showImage(img)
		return img

	def readEachChannel(self, path, show=False):
		img = self.readColorImage(path)
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		rgb = cv2.split(img)
		if show :
			plt.subplot("141"); plt.title("Original"); plt.imshow(img)
			plt.subplot("142"); plt.title("R");  plt.imshow(rgb[0])
			plt.subplot("143"); plt.title("G");  plt.imshow(rgb[1])
			plt.subplot("144"); plt.title("B");  plt.imshow(rgb[2])
			plt.show()
		return rgb

	def histogramGrayScale(self, img, show=False):
		if show :
			plt.subplot("211"); plt.title("Original"); plt.imshow(img,'gray')
			plt.subplot("212"); plt.title("Histogram"); plt.hist(img.ravel(), 256, [0, 256])
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
		if show :
			cv2.imshow("img", img)
			cv2.waitKey(0)
			cv2.destroyAllWindows()
		return img

	def createGrayScaleRandomImage(self, size, show=False):
		img = np.ones(size, dtype=np.float64)
		cv2.randu(img, 0, 255)
		if show :
			cv2.imshow("img", img)
			cv2.waitKey(0)
			cv2.destroyAllWindows()
		return img

	def createColorRandomImage(self, size, show=False):
		new_size = (size[0],size[1],3)
		img = np.ones(new_size, dtype=np.uint8)
		bgr = cv2.split(img)
		for i in range(3):
			cv2.randu(bgr[i], 0, 255)
		img = cv2.merge(bgr)
		if show :
			cv2.imshow("img", img)
			cv2.waitKey(0)
			cv2.destroyAllWindows()
		return img

	def createGrayScaleRandomNormallyImage(self, size, show=False):
		img = np.ones(size, dtype=np.uint8)
		cv2.randn(img, 127, 40)
		histr = cv2.calcHist([img], [], None, [255], [0, 255])
		if show :
			plt.plot(histr)
			plt.xlim([0, 256])
			plt.show()
		return img
		
	def createColorRandomNormallyImage(self, size, show=False):
		new_size = (size[0],size[1],3)
		img = np.ones(new_size, dtype=np.uint8)
		bgr = cv2.split(img)
		cv2.randn(bgr[0], 127, 40)
		cv2.randn(bgr[1], 127, 40)
		cv2.randn(bgr[2], 127, 40)
		img = cv2.merge(bgr)
		if show :
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
		img2 = np.where( img2 > 255 , 255 , img2 )
		img2 = np.asarray(img2, dtype=np.uint8)
		if show :
			plt.subplot("121"); plt.title("IMG 1"); plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
			plt.subplot("122"); plt.title("IMG 2"); plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
			plt.show()
		return img2

	def sumImages(self, img1, img2, scalar1=1, scalar2=1, show=False):
		img1 = np.asarray(img1, dtype=np.float64)
		img2 = np.asarray(img2, dtype=np.float64)
		img3 = scalar1*img1 + scalar2*img2
		img3 = np.where( img3 > 255 , 255 , img3 )
		
		img1 = np.asarray(img1, dtype=np.uint8)
		img2 = np.asarray(img2, dtype=np.uint8)
		img3 = np.asarray(img3, dtype=np.uint8)
		
		if show :
			plt.subplot("131"); plt.title("IMG 1"); plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
			plt.subplot("132"); plt.title("IMG 2"); plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
			plt.subplot("133"); plt.title("IMG 3"); plt.imshow(cv2.cvtColor(img3, cv2.COLOR_BGR2RGB))
			plt.show()

		return img3

	def maxImages(self, img1, img2, show=False):
		img3 = cv2.max(img1, img2)

		if show :
			plt.subplot("131"); plt.title("IMG 1"); plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
			plt.subplot("132"); plt.title("IMG 2"); plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
			plt.subplot("133"); plt.title("IMG 3"); plt.imshow(cv2.cvtColor(img3, cv2.COLOR_BGR2RGB))
			plt.show()

		return img3

	def absdiffImages(self, img1, img2, show=False):
		img3 = cv2.absdiff(img1, img2)
		
		if show :
			plt.subplot("131"); plt.title("IMG 1"); plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
			plt.subplot("132"); plt.title("IMG 2"); plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
			plt.subplot("133"); plt.title("IMG 3"); plt.imshow(cv2.cvtColor(img3, cv2.COLOR_BGR2RGB))
			plt.show()

		return img3

	def addNoiseinImage(self, img, show=False):
		noise = self.createColorRandomImage(img.shape)
		img2 = self.sumImages(img, noise)
		
		if show :
			plt.subplot("121"); plt.title("IMG 1"); plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
			plt.subplot("122"); plt.title("IMG 2"); plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
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

		if show :
			plt.subplot("121"); plt.title("IMG 1"); plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
			plt.subplot("122"); plt.title("IMG 2"); plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
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

		if show :
			plt.subplot("121"); plt.title("IMG 1"); plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
			plt.subplot("122"); plt.title("IMG 2"); plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
			plt.show()

		return img2

	def andImages(self, img1, img2, show=False):
		and_img = img1 & img2
		
		if show :
			plt.subplot("131"); plt.title("IMG 1"); plt.imshow(img1, 'gray')
			plt.subplot("132"); plt.title("IMG 2"); plt.imshow(img2, 'gray')
			plt.subplot("153"); plt.title("AND"); plt.imshow(and_img, 'gray')
			plt.show()
		
		return and_img

	def orImages(self, img1, img2, show=False):
		or_img = img1 | img2
		
		if show :
			plt.subplot("131"); plt.title("IMG 1"); plt.imshow(img1, 'gray')
			plt.subplot("132"); plt.title("IMG 2"); plt.imshow(img2, 'gray')
			plt.subplot("153"); plt.title("OR"); plt.imshow(or_img, 'gray')
			plt.show()
		
		return or_img

	def orImages(self, img1, img2, show=False):
		not_img = ~img
		
		if show :
			plt.subplot("131"); plt.title("IMG 1"); plt.imshow(img1, 'gray')
			plt.subplot("132"); plt.title("IMG 2"); plt.imshow(img2, 'gray')
			plt.subplot("153"); plt.title("NOT"); plt.imshow(not_img, 'gray')
			plt.show()
		
		return not_img

	def logTransformImg(self, img, scalar=1, norm=0, show=False):
		img1 = np.asarray(img, dtype=np.float64)
		img1 = img1/255
		img2 = np.ones(img.shape, np.float64)
		img2 = scalar * np.log(1 + img1)

		if norm :
			cv2.normalize(img2, img2, 1, 0, cv2.NORM_MINMAX)
		else :
			img2 = np.where( img2 > 1 , 1 , img2 )

		img2 = img2 * 255
		img2 = np.asarray(img2, dtype=np.uint8)

		if show :
			cv2.imshow("img", img)
			cv2.imshow("img2", img2)
			cv2.waitKey(0)
			cv2.destroyAllWindows()

		return img2

	def gammaTransformImg(self, img, exp=1, show=False):
		img1 = np.asarray(img, dtype=np.float64)
		img1 = img1/255
		img1 = img1 ** exp # np.power(img1,exp)
		img1 = img1 * 255
		img1 = np.asarray(img1, dtype=np.uint8)

		if show :
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
			n = cv2.getTrackbarPos("n", "img2")/10
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
		new_size = (size[0],size[1])
		return cv2.resize(img, new_size, 0, 0, cv2.INTER_LINEAR)

	def thresholdingImage(self, img, lim=(150,240), show=False):
		img2 = np.copy(img)
		img2 = np.where( (lim[0] < img2) & (img2 < lim[1]), img2 , 0 )

		if show :
			cv2.imshow("img", img2)
			cv2.waitKey(0)
			cv2.destroyAllWindows()

		return img2

	def bitSlicingImage(self, img, slices, show=False):
		mask = 0
		for bit in slices:
			mask = mask | (1<<bit)

		img2 = img & mask

		if show :
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

		if show :
			cv2.imshow("img", img)
			cv2.imshow("img2", img2)
			cv2.imshow("hist", hist)
			cv2.imshow("hist2", hist2)
			cv2.waitKey(0)
			cv2.destroyAllWindows()

		return img2

	def localHistogramEqualization(self, img, size=(1,1), show=False):
		img2 = np.zeros(img.shape, img.dtype)
		hsize, vsize = size

		for x in range(hsize, img.shape[0] - hsize):
			for y in range(vsize, img.shape[1] - vsize):
				cv2.equalizeHist(img[y - vsize: y + vsize][x - hsize: x + hsize],img2[y - vsize: y + vsize][x - hsize: x + hsize])

		if show :
			cv2.imshow("img", img)
			cv2.imshow("img2", img2)
			cv2.waitKey(0)
			cv2.destroyAllWindows()

		return img2
