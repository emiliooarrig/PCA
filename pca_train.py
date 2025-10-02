# Este programa es para entrenar al 
# modelo que será guardado en el folder 
# de "models".

#Funciones necesarias:

# importar y usar pca_process para procesar 
# las imagenes y conseguir los vectores

# train_model  el resultado es guardado 
# en "models"

import numpy as np
import cv2
import os

import pca_process as pca

def borrar_normdata(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        if os.path.isfile(file_path):  # make sure it's a file
            os.remove(file_path)

def main():
    print("TRAINING MODEL")
    borrar_normdata("normalized_data")
    pca.proc_matrix("data")

if __name__ == "__main__":
    main()
