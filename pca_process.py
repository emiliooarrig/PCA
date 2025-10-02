#En este programa se procesan las imagenes
#para poder ser usadas en test o en train

#Funciones necesarias:
#--------------------------------------------------
#pro_image procesa imagenes a vectores 
# (los pasa a escala de grises, reduce su 
# tamaño y lo pasa a vectores)

#vect_process toma los vectores de 
# una persona y hace el analisis de 
# componentes
#--------------------------------------------------

#funcion regresa imagenes escala de grises (entra imagen 60*90)
#

#funcion regresa matriz de vectores de grises (entra lista de imagenes grises 60*90)
#

#funcion regresa componente principal como vector (entra matriz de valores de grises)
#

#funcion regresa componente principal (entra lista de imagenes png de 60x90)
#   pasa imagenes a escala de grises
#   lo pasa a vectores
#   hace analisis de componentes
#   encuentra componente principal
#   imprime y regresa componente principal como vector



import numpy as np
import cv2
import os

#Normaliza las imagenes a escala de grises
def normalizar(filename):
    # Input & output
    input_path = os.path.join("data", filename)
    output_path = os.path.join("normalized_data", filename)

    # Lee colores
    img = cv2.imread(input_path)
    if img is None:
        raise FileNotFoundError(f"Image {input_path} not found!")
    
    # Convierte a gris y transforma la imagen a 60*90 si es necesario
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_resized = cv2.resize(gray, (60, 90))

    # Checa si el folder existe y guarda la imagen
    os.makedirs("normalized_data", exist_ok=True)
    cv2.imwrite(output_path, gray_resized)

    print(f"Imagen normalizada guardada en {output_path}")

#Pasa las imagenes grises a matriz numpy
def imag_Matriz(filename):
    # Ruta de la imagen normalizada
    input_path = os.path.join("normalized_data", filename)

    # Leer la imagen en escala de grises
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise FileNotFoundError(f"Image {input_path} not found!")

    # Convertir a matriz NumPy normalizada (0.0–1.0)
    matrix = img.astype(np.float32) / 255.0

    print(f"✅ Imagen {filename} convertida a matriz con forma {matrix.shape}")
    return matrix

#Procesa las matrices (lista de matrices) a 1 sola matriz

#Funcion principal donde se lamman a todas las funciones de procesamiento de imagenes (regresa la componente principal de la persona)
def main():
    rang= len(os.listdir("data"))
    img= 60*90

    # Preallocate empty matrix
    matrix = np.zeros((rang, img), dtype=np.float32)
    i=0
    for filename in os.listdir("data"):

        # Skip non-image files
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            print(f"Processing {filename} ...")
            normalizar(filename) #normaliza la imagen a gris

            #prepara las matrices para ser procesadas
            row = imag_Matriz(filename).flatten()# pasa la imagen gris a matriz y lo convierte en fila de la matriz de "persona"
            matrix[i, :] = row  
            i=i+1 
    print(matrix)
    



if __name__ == "__main__":
    main()


