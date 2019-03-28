import library as lib

# Lendo a imagem
img = lib.readColorImage(path='chips.png')

# Transformando para HSV
hsv = lib.bgr2hsv( img )
lib.plotImageHSV(hsv)

# Separando os canais
hsv_split = lib.splitChanels(hsv)

shape = img.shape
blue = lib.createImgTom(shape[0],shape[1], 24)
green = lib.createImgTom(shape[0],shape[1], 65)
yellow = lib.createImgTom(shape[0],shape[1], 120)
orage = lib.createImgTom(shape[0],shape[1], 168)
red = lib.createImgTom(shape[0],shape[1], 200)

white = lib.createImgTom(shape[0],shape[1], 255)

# H -> Cor
# S -> Intensidade
# V -> Brilho

colors = { 	'Azul' : 24 	, 
			'Verde' : 70 	,
			'Amarelo' : 130 ,
			'Laranja' : 160	,
			'Vermelho': 160
		}

# Histograma do canal H
a = lib.plt.hist(hsv_split[0].ravel(), bins=256, range=(0, 256), fc='k', ec='k')
a = lib.np.array(a)

lib.plotImage(lib.scaleImage2_uchar(a))

# Azul 24
# Verde 70
# Amarelo 130
# Laranja 190
# Vermelho

filt = lib.sat(hsv_split[0], 24, 240)
lib.plotImage(filt)	
m = hsv_split[0] & filt 
hsv2 = lib.mergeChanels(hsv_split[0], hsv_split[1], filt)
lib.plotImageHSV(hsv2)


'''
filt = lib.sat(hsv_split[0], 24, 5)



lib.plotImage(filt, 'filtro')

m = hsv_split[2] & filt 
hsv2 = lib.mergeChanels(hsv_split[0], hsv_split[1], 255-m)

lib.plotImageHSV(hsv2)


a = lib.plt.hist(hsv_split[0].ravel(),256,[0,256])
lib.plt.show()
'''