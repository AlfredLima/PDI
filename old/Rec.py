import cv2

webcam = cv2.VideoCapture(0)  #instancia o uso da webcam
count = 1

path = "~/Documents/Softwares/opencv/data/haarcascades/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(path)

while True:
	s, imagem = webcam.read() #pega efeticamente a imagem da webcam
	imagem = cv2.flip(imagem,180) #espelha a imagem

	gray = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
	faces = faceCascade.detectMultiScale(
	    gray,
	    scaleFactor=1.1,
	    minNeighbors=5,
	    minSize=(30, 30)
	)

	# Desenha um retângulo nas faces detectadas
	for (x, y, w, h) in faces:
		cv2.rectangle(imagem, (x, y), (x+w, y+h), (0, 255, 0), 2)

	# Desenha um retângulo nas faces detectadas
	cv2.imshow('Video', imagem) #mostra a imagem captura na janela

	#o trecho seguinte é apenas para parar o código e fechar a janela
	t = cv2.waitKey(1) 
	if t == ord('q'):
		break

webcam.release() #dispensa o uso da webcam
cv2.destroyAllWindows() #fecha todas a janelas abertas