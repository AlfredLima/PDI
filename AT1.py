import cv2

webcam = cv2.VideoCapture(0)  #instancia o uso da webcam
count = 1

while True:
	s, imagem = webcam.read() #pega efeticamente a imagem da webcam
	imagem = cv2.flip(imagem,180) #espelha a imagem

	# Desenha um retângulo nas faces detectadas
	cv2.imshow('Video', imagem) #mostra a imagem captura na janela

	#o trecho seguinte é apenas para parar o código e fechar a janela
	t = cv2.waitKey(1) 
	if t == ord('q'):
		break
	elif t == ord('s'):
		cv2.imwrite( "PDI_" + str(count) + ".jpg", imagem )
		count = count + 1
		print('Salvando')

webcam.release() #dispensa o uso da webcam
cv2.destroyAllWindows() #fecha todas a janelas abertas