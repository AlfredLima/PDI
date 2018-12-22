import pdi

library = pdi.PDI()

path = "AT2/lena.png"
path1 = "AT2/baboon.png"

#library.showImage(a)

'''		Read Image
library.readGrayScaleImage(path, True)
library.readColorImage(path, True)
library.readEachChannel(path, True)
'''

'''		Histogram
library.histogramGrayScale(path, True)
library.histogramColor(path, True)
'''

'''		Create Image
size = (512,512)
library.createGrayScaleImage(size,scale=0,show=True)
library.createGrayScaleRandomImage(size,show=True)
library.createColorRandomImage(size,show=True)
library.createGrayScaleRandomNormallyImage(size,show=True)
library.createColorRandomNormallyImage(size)
'''

'''		Operation
'''
#library.addScalarImage(path,scalar=100,show=True)
#library.sumImages(path, path1, scalar1=0.5, scalar2=0.5, show=True)
#library.maxImages(path, path1, show=True)
#library.absdiffImages(path, path1, show=True)

img = library.readColorImage(path)
img = library.readGrayScaleImage(path)
#library.addNoiseinImage(img, show=True)
#library.addSaltPepperinImage(img, show=True)
#library.logTransformImg(img=img,show=True,norm=0)
#library.gammaTransformImg(img=img, exp=2, show=True)
library.testGammaTransform(img)
#library.thresholdingImage(img, show=True)
#library.bitSlicingImage(img, slices=[7,6,5,4], show=True)
#library.localHistogramEqualization(img, show=True, size=(50,100))