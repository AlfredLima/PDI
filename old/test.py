import library as lib

img1 = lib.createWhiteDisk(height=400, width=400)
img2 = lib.createBlackDisk(height=400, width=400)
img3 = lib.createBlackRing()
img4 = lib.createWhiteRing()
img5 = lib.create2DGaussian(height=400, width=400, mx = 200, my = 200)
img6 = lib.createSineImage(height=400, width=400, freq=500, theta=45)
img7 = lib.createSineImage(height=400, width=400, freq=50, theta=-45)
img8 = lib.sumImage(img6,img7)

#lib.plotImage(img1)
#lib.plotImage(img2)
#lib.plotImage(img3)
#lib.plotImage(img4)

lib.plotImage(img8)

a,b = lib.FFTinImage(img8)
lib.plotImage( lib.magnitude(b) )

