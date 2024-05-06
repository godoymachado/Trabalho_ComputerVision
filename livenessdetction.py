import cv2
import numpy as np
import time

# Inicializar a câmera
cap = cv2.VideoCapture(0)

# Carregar o classificador de faces e o modelo de reconhecimento
face_cascade = cv2.CascadeClassifier('D:\Computer_Vision\haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('D:\Computer_Vision\classificadorLbph.yml')  # Substitua pelo caminho do seu arquivo de modelo treinado

# Ler o primeiro quadro
ret, frame1 = cap.read()
gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
gray1 = cv2.GaussianBlur(gray1, (21, 21), 0)

last_motion_time = 0  # Tempo da última detecção de movimento
display_duration = 3  # Duração para mostrar a mensagem "Pass" (em segundos)

while True:
    ret, frame2 = cap.read()
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.GaussianBlur(gray2, (21, 21), 0)

    delta_frame = cv2.absdiff(gray1, gray2)
    threshold_frame = cv2.threshold(delta_frame, 25, 255, cv2.THRESH_BINARY)[1]
    threshold_frame = cv2.dilate(threshold_frame, None, iterations=2)
    
    faces = face_cascade.detectMultiScale(gray2, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    motion_detected = False

    for (x, y, w, h) in faces:
        roi_gray = gray2[y:y+h, x:x+w]
        id_, conf = recognizer.predict(roi_gray)  # Reconhecer quem é a pessoa na imagem
        if conf >= 50 and conf <= 100:  # Ajuste esses valores conforme necessário
            cv2.rectangle(frame2, (x, y), (x+w, y+h), (0, 255, 0), 2)
            motion_detected = True
            cv2.putText(frame2, f'ID: {id_}', (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            last_motion_time = time.time()  # Atualizar o tempo da última detecção de movimento

    if (time.time() - last_motion_time) < display_duration:
        cv2.putText(frame2, 'Chupa', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    else:
        cv2.putText(frame2, 'Not Pass', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imshow('Frame', frame2)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

    gray1 = gray2  # Atualizar o quadro anterior para a próxima iteração

cap.release()
cv2.destroyAllWindows()
