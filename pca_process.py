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

    print(f" Imagen {filename} convertida a matriz con forma {matrix.shape}")
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
    print("\nMatriz U (Ortogonal):")
    print(U)
    #VALORES SINGULARES
    print("\nMatriz Σ (valores singulares):")
    print(Sigma)
    #ORIGINAL X TRANSPUESTA
    print("\nMatriz V^T (Transpuesta):")
    print(VT)

    # NUEVO: Calcular varianzas explicadas y rango
    varianzas = (S ** 2) / np.sum(S ** 2)
    print("\nVarianzas explicadas:")
    print(varianzas)
    r = np.sum(S > 1e-10)
    print(f"Rango de la matriz (r): {r}")

    # Retornamos las tres matrices y valores singulares para pasos posteriores
    return U, Sigma, VT, S, varianzas


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
    u,sig,vt,S,varianzas=matriz_svd(matrix) 
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
    
    # Imprime la matriz centrada
    print("\nMatriz Centrada (matrix_centered):")
    print(matrix_centered) 
    
    # Aplicar SVD
    print("\n" + "=" * 60)
    print("SVD")
    print("=" * 60)
    u, sig, vt, S, varianzas = matriz_svd(matrix_centered)
    
    # Proyectar al espacio PCA con k componentes
    proy = np.dot(matrix_centered, vt[:k, :].T)
    
    print(f"\n Entrenamiento completado con {k} componentes principales")
    # Imprimir las proyecciones Z
    print("Proyecciones Z. ")
    print(proy)
    
    # Reconstruccion y errores

    # Con esta linea reconstruimos el X_hats
    X_hat = np.dot(proy, vt[:k, :]) + mean_face
    # Imprime la matriz reconstruida
    print("\nMatriz Reconstruida (X_gorro):")
    print(X_hat)
    
    return mean_face, vt, proy


def predecir(ruta_imagen_test, mean_face, vt, proy_entrenamiento, k=5, umbral=0.5):
    """
    Predice si la imagen de prueba pertenece al mismo sujeto
    Regresa: distancia_minima, es_aceptado, error_reconstruccion_test
    """
    print("\n" + "=" * 60)
    print("INICIANDO PREDICCIÓN")
    print("=" * 60)
    
    # Lee y procesa imagen de prueba
    print(f" Procesando imagen: {ruta_imagen_test} \n")
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
    
    print(f"Proyección de prueba: {proyeccion_test}")

    # CÁLCULO Y IMPRESIÓN DEL ERROR DE RECONSTRUCCIÓN DE PRUEBA 
    # 1. Reconstruir el vector de prueba (x_hat_test)
    x_hat_test = np.dot(proyeccion_test, vt[:k, :]) + mean_face
    
    # 2. Calcular el error de reconstrucción (norma Euclidiana)
    error_reconstruccion_test = np.linalg.norm(vector_test - x_hat_test)
    
    # Imprimir el error de reconstruccion de prueba
    print(f"\nError de Reconstrucción (X_gorro): {error_reconstruccion_test:.6f}")
    
    # Buscar el vecino más cercano usando distancia euclidiana
    distancia_minima = float('inf')
    
    print("\n Distancias (vecino más cercano):")
    for i in range(len(proy_entrenamiento)):
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
    
    return distancia_minima, es_aceptado, error_reconstruccion_test


if __name__ == "__main__":    

    # Normalizar las imagenes antes de empezar el entrenamiento
    for i in range(0, 5):
        nombre_foto = "p" + str(i + 1) + ".png"
        normalizar(nombre_foto)
        nombre_foto = ""
    
    # FASE 1: ENTRENAMIENTO
    k_componentes = 5
    
    mean_face, vt, proy_entrenamiento = entrenar(folder="normalized_data", k=k_componentes)
    
    
    # FASE 2: PRUEBA
    imagen_test = "prueba-similar.png"
    # Ajustar el umbral dependiendo la prueba
    umbral = 0.9  
    
    distancia, aceptado, error_test = predecir(
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
    print(f"Error de Reconstrucción (Test): {error_test:.6f}")
    print(f"Decisión: {'ACEPTADO' if aceptado else 'RECHAZADO'}")