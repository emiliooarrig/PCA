# En este programa se procesan las imagenes
# para poder ser usadas en test o en train

# Funciones necesarias:
# --------------------------------------------------
# pro_image procesa imagenes a vectores
# (los pasa a escala de grises, reduce su
# tamaño y lo pasa a vectores)

# vect_process toma los vectores de
# una persona y hace el analisis de
# componentes
# --------------------------------------------------

# funcion regresa imagenes escala de grises (entra imagen 60*90)
#

# funcion regresa matriz de vectores de grises (entra lista de imagenes grises 60*90)
#

# funcion regresa componente principal como vector (entra matriz de valores de grises)
#

# funcion regresa componente principal (entra lista de imagenes png de 60x90)
#   pasa imagenes a escala de grises
#   lo pasa a vectores
#   hace analisis de componentes
#   encuentra componente principal
#   imprime y regresa componente principal como vector


import numpy as np
import cv2
import os


# Normaliza las imagenes a escala de grises
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


# Pasa las imagenes grises a matriz numpy
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


def matriz_svd(matriz):
    # Convertimos la matriz a un arreglo numpy
    A = np.array(matriz)

    # Aplicamos SVD
    U, S, VT = np.linalg.svd(A)

    # S contiene solo los valores singulares, no la matriz diagonal completa
    # Para convertirlo a una matriz diagonal del mismo tamaño que A:
    Sigma = np.zeros((A.shape[0], A.shape[1]))
    np.fill_diagonal(Sigma, S)

    print("Matriz original:")
    print(A)
    #VECTORES PROPIOS
    print("\nMatriz U:")
    print(U)
    #VALORES SINGULARES
    print("\nMatriz Σ (valores singulares):")
    print(Sigma)
    #ORIGINAL X TRANSPUESTA
    print("\nMatriz V^T:")
    print(VT)

    # Retornamos las tres matrices
    return U, Sigma, VT


# Funcion principal donde se lamman a todas las funciones de
# procesamiento de imagenes (regresa la componente principal)
def proc_matrix(folder):
    
    rang = len(os.listdir(folder))
    img = 60 * 90

    # Preallocate empty matrix
    matrix = np.zeros((rang, img), dtype=np.float32)
    i = 0
    for filename in os.listdir(folder):

        # Skip non-image files
        if filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
            print(f"Processing {filename} ...")
            normalizar(filename)  # normaliza la imagen a gris

            # prepara las matrices para ser procesadas
            # pasa la imagen gris a matriz y lo convierte en fila de la matriz de "persona"
            row = imag_Matriz(filename).flatten()
            matrix[i, :] = row
            i = i + 1

    # Mandar todas las matrices al algoritmo de SVD
    print ("SVD-----------------------------------------")
    u,sig,vt=matriz_svd(matrix) #ahora podemos usar "svd para conseguir u, sigma y/o vt usando svd(#)"
    matrix_centered = matrix - np.mean(matrix, axis=0)
    #proyeciones principales (producto punto de matrix*s)= vector resultante que se regresa.
    k=5
    proy=np.dot(matrix_centered, vt[:k, :].T)
    print("PROYECCION")
    print (proy)
    pc1 = proy[0, :]  # vector de tamaño (n_muestras,)
    print("PRINCIPAL")
    print(pc1)
    return pc1


# ============================================================================
# NUEVAS FUNCIONES PARA ENTRENAMIENTO Y PRUEBA
# ============================================================================

# Funcion de entrenamiento: procesa 5 imagenes de entrenamiento
# y regresa las proyecciones y componentes principales
def entrenar(folder="normalized_data", k=5):
    """
    Entrena el sistema con 5 imagenes de la carpeta normalized_data
    Regresa: mean_face, vt (componentes principales), proyecciones
    """
    print("=" * 60)
    print("INICIANDO ENTRENAMIENTO")
    print("=" * 60)
    
    img = 60 * 90
    vectores = []
    contador = 0
    
    # Procesar solo las primeras 5 imagenes de la carpeta
    print(f"\nProcesando imagenes de entrenamiento desde: {folder}")
    
    for filename in os.listdir(folder):
        if contador >= 5:
            break
            
        if filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
            ruta_imagen = os.path.join(folder, filename)
            print(f"  - {filename}")
            
            # Lee la imagen en escala de grises
            img_data = cv2.imread(ruta_imagen, cv2.IMREAD_GRAYSCALE)
            if img_data is None:
                continue
            
            # Convertir a vector normalizado (0.0–1.0)
            vector = img_data.astype(np.float32) / 255.0
            row = vector.flatten()
            
            vectores.append(row)
            contador += 1
    
    # Convertir a matriz numpy
    matrix = np.array(vectores)
    
    print(f"\n Total de imágenes de entrenamiento: {matrix.shape[0]}")
    
    # Calcular cara promedio y centrar datos
    mean_face = np.mean(matrix, axis=0)
    matrix_centered = matrix - mean_face
    
    # Aplicar SVD
    print("\n Aplicando SVD...")
    u, sig, vt = matriz_svd(matrix_centered)
    
    # Proyectar al espacio PCA con k componentes
    proy = np.dot(matrix_centered, vt[:k, :].T)
    
    print(f"\n Entrenamiento completado con {k} componentes principales")
    print("PROYECCIONES DE ENTRENAMIENTO:")
    print(proy)
    
    return mean_face, vt, proy


# Funcion de prediccion: usa vecino mas cercano para identificar sujeto
def predecir(ruta_imagen_test, mean_face, vt, proy_entrenamiento, k=5, umbral=0.5):
    """
    Predice si la imagen de prueba pertenece al mismo sujeto
    Regresa: distancia_minima, es_aceptado
    """
    print("\n" + "=" * 60)
    print("INICIANDO PREDICCIÓN")
    print("=" * 60)
    
    # Lee y procesa imagen de prueba
    print(f" Procesando imagen: {ruta_imagen_test}")
    img = cv2.imread(ruta_imagen_test)
    if img is None:
        raise FileNotFoundError(f"Image {ruta_imagen_test} not found!")
    
    # Convierte a gris y transforma la imagen a 60*90
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_resized = cv2.resize(gray, (60, 90))
    
    # Convertir a vector normalizado
    vector_test = gray_resized.astype(np.float32) / 255.0
    vector_test = vector_test.flatten()
    
    # Centrar y proyectar al espacio PCA
    vector_centrado = vector_test - mean_face
    proyeccion_test = np.dot(vector_centrado, vt[:k, :].T)
    
    print(f"Imagen proyectada al espacio PCA")
    print(f"Proyección de prueba: {proyeccion_test}")
    
    # Buscar el vecino más cercano usando distancia euclidiana
    distancia_minima = float('inf')
    
    print("\n Calculando distancias (vecino más cercano):")
    for i in range(len(proy_entrenamiento)):
        # Distancia euclidiana
        dist = np.linalg.norm(proyeccion_test - proy_entrenamiento[i])
        print(f"  - Imagen entrenamiento {i+1}: distancia = {dist:.4f}")
        
        if dist < distancia_minima:
            distancia_minima = dist
    
    # Decisión basada en umbral
    es_aceptado = distancia_minima <= umbral
    
    print("\n" + "=" * 60)
    print("RESULTADO")
    print("=" * 60)
    print(f"Distancia mínima: {distancia_minima:.4f}")
    print(f"Umbral de decisión: {umbral:.4f}")
    
    if es_aceptado:
        print(f"ACEPTADO - La imagen pertenece al mismo sujeto")
    else:
        print(f"RECHAZADO - La imagen NO pertenece al mismo sujeto")
    
    return distancia_minima, es_aceptado


# ============================================================================
# EJEMPLO DE USO
# ============================================================================

if __name__ == "__main__":
    
    # FASE 1: ENTRENAMIENTO
    # Usa las 5 imagenes de la carpeta normalized_data
    
    k_componentes = 5
    
    mean_face, vt, proy_entrenamiento = entrenar(folder="normalized_data", k=k_componentes)
    
    
    # FASE 2: PRUEBA
    # Probar con una imagen nueva
    
    imagen_test = "prueba-mala.png"
    umbral = 0.5  # Ajustar según necesidad
    
    distancia, aceptado = predecir(
        imagen_test, 
        mean_face, 
        vt, 
        proy_entrenamiento, 
        k=k_componentes, 
        umbral=umbral
    )
    
    print("\n" + "=" * 60)
    print("RESUMEN FINAL")
    print("=" * 60)
    print(f"Distancia mínima: {distancia:.4f}")
    print(f"Decisión: {'ACEPTADO' if aceptado else 'RECHAZADO'}")