import library as lib

# Lendo a imagem
img = lib.readColorImage(path='chips.png')

# Transformando para HSV
hsv = lib.bgr2hsv( img )
lib.plotImageHSV(hsv)

# Separando os canais
hsv_split = lib.splitChanels(hsv)

# H -> Cor
# S -> Intensidade
# V -> Brilho

colors = { 	'Azul' : 24 	, 
			'Verde' : 70 	,
			'Amarelo' : 130 ,
			'Laranja' : 170	,
			'Vermelho': 180
		}


for c in colors:
	filt = lib.sat(hsv_split[0], colors[c], 240)
	lib.plotImage(filt, 'Filtro para ' + c)	
	m = hsv_split[0] & filt 
	hsv2 = lib.mergeChanels(hsv_split[0], hsv_split[1], filt)
	rgb = lib.hsv2bgr(hsv2)
	lib.plotImage(rgb, c)