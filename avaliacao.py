import cv2
import os
import numpy as np
import pandas as pd

# Carregar o classificador de faces
detectorFace = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Carregar modelos
eigen_recognizer = cv2.face.EigenFaceRecognizer_create()
eigen_recognizer.read('D:\Computer_Vision\classificadorEigen.yml')

fisher_recognizer = cv2.face.FisherFaceRecognizer_create()
fisher_recognizer.read('D:\Computer_Vision\classificadorfisher.yml')

lbph_recognizer = cv2.face.LBPHFaceRecognizer_create()
lbph_recognizer.read('D:\Computer_Vision\classificadorLbph.yml')

# Caminho para o diretório com imagens de teste
test_dir = 'D:\Computer_Vision\imagens'
test_images = [os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.endswith('.jpg')]

def evaluate_model(recognizer, images, target_size=(220, 220)):
    correct = 0
    total = 0
    for image_path in images:
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detectorFace.detectMultiScale(gray, scaleFactor=1.1, minSize=(100,100))

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_gray = cv2.resize(roi_gray, target_size)  # Redimensionar para o tamanho esperado pelo modelo
            label, confidence = recognizer.predict(roi_gray)
            expected_label = int(os.path.split(image_path)[-1].split('.')[1])  # Assuming filename format: pessoa.id.number.jpg
            if label == expected_label:
                correct += 1
            total += 1

    accuracy = (correct / total) * 100 if total > 0 else 0
    return accuracy

# Avaliar cada modelo
eigen_accuracy = evaluate_model(eigen_recognizer, test_images)
fisher_accuracy = evaluate_model(fisher_recognizer, test_images)
lbph_accuracy = evaluate_model(lbph_recognizer, test_images)

# Criar uma tabela com os resultados
results = pd.DataFrame({
    "Modelo": ["Eigenfaces", "Fisherfaces", "LBPH"],
    "Acurácia (%)": [eigen_accuracy, fisher_accuracy, lbph_accuracy]
})

print(results)
