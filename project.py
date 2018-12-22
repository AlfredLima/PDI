import face_recognition
import cv2
import time
green = (0, 255, 0)
red = (0, 0, 255)

webcam = cv2.VideoCapture(0)  #instancia o uso da webcam

while True:
	a = time.time()
	s, imagem = webcam.read() #pega efeticamente a imagem da webcam
	imagem = cv2.flip(imagem,180) #espelha a imagem
	small_frame = cv2.resize(imagem, (0, 0), fx=0.25, fy=0.25)
	face_locations = face_recognition.face_locations(small_frame)	

	for (top, right, bottom, left) in face_locations:
		top = 4*top
		right = 4*right
		bottom = 4*bottom
		left = 4*left
		cv2.rectangle(imagem, (left, top), (right,bottom), green, 10)

	
	# Desenha um retângulo nas faces detectadas
	cv2.imshow('Video', imagem) #mostra a imagem captura na janela

	#o trecho seguinte é apenas para parar o código e fechar a janela
	t = cv2.waitKey(1) 
	if t == ord('q'):
		break
	
	aa = time.time()
	print( aa - a )


webcam.release() #dispensa o uso da webcam
cv2.destroyAllWindows() #fecha todas a janelas abertas