import cv2
import os 
from cycler import L
import numpy as np

eigenface = cv2.face.EigenFaceRecognizer.create()
fisherface = cv2.face.FisherFaceRecognizer.create()
lbph =cv2.face.LBPHFaceRecognizer.create()

#retorna imagens de ids especificos, é um aprendizado supervisionado
def getImagemComid():
    caminhos = [os.path.join('imagens', f)for f in os.listdir('imagens')]
    #print(path)
    faces = []
    ids = []
    for caminhoImagem in caminhos:
        imagemFace =cv2.cvtColor(cv2.imread(caminhoImagem), cv2.COLOR_BGRA2GRAY)
        #capiturando os ids
        id =int(os.path.split(caminhoImagem)[-1].split('.')[1])
        ids.append(id)
        faces.append(imagemFace)

        #cv2.imshow('Face',imagemFace)
        #cv2.waitKey(0)
    return np.array(ids), faces
ids, faces = getImagemComid()
print ('Treinando')

#treinando com eigenfaces
eigenface.train(faces, ids)
eigenface.write('classificadorEigen.yml')
#treinando com Fisher
fisherface.train(faces, ids)
fisherface.write('classificadorfisher.yml')
#treinando com LBPH
lbph.train(faces, ids)
lbph.write('classificadorLbph.yml')
print(faces)
