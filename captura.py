import cv2
#CLassificador de faces
classifier = cv2.CascadeClassifier('D:\Computer_Vision\haarcascade_frontalface_default.xml')
camera = cv2.VideoCapture(0)
# Salvando as amostras e ao apertar uma tecla as as fotos são salvas
amostra = 1
numeroAmostras=30
id = input('Qual é o numero da pessoa:')
largura, altura = 220,220
print('Tirando as fotos')

while (True):
    conectado, imagem = camera.read()
    if not conectado:
        continue  # Skip the rest of the loop iteration if the camera failed to capture a frame

    imagemCinza = cv2.cvtColor(imagem, cv2.COLOR_BGRA2GRAY)
    facesDetectadas = classifier.detectMultiScale(imagemCinza, scaleFactor=1.5, minSize=(100,100))

    # Adicionando os quadradinhos para identificação de faces com posição no eixo x e y, junto com largura e altura da face
    for (x,y,l,a) in facesDetectadas:
        cv2.rectangle(imagem, (x,y),(x + l,y + a),(0,0,255),2)
        # ao apertar a tecla k a imagem será salva
        if cv2.waitKey(1) & 0xff == ord('k'):
            imagemface = cv2.resize(imagemCinza[y:y + a, x:x + l], (largura, altura))
            if cv2.imwrite('imagens/pessoa.' + str(id) + '.' + str(amostra) + '.jpg', imagemface):
                print('[foto ]' + str(amostra) + ' capturada com maestria')
            else:
                print('Erro ao salvar a imagem')
            amostra += 1

    cv2.imshow('Face', imagem)
    #cv2.waitKey(1)
    if (amostra >= numeroAmostras + 1):
        break
camera.release()
cv2.destroyAllWindows()
