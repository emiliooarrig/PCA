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

def main():
    print("hello world!")
    pca.proc_matrix("data")
    
if __name__ == "__main__":
    main()
