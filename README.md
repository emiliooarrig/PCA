PROYECTO MEDIO TERMINO
ALGEBRA LINEAL AVANZADA

EMILIO GUZMAN
FERNANDO REVILLA

SISTEMA RECONOCIMIENTO FACIAS USANDO ANALISIS DE COMPONENTES PRINCIPALES

Pasos propuestos:
    Recolección de datos

        Reunir carpetas por identidad: data/person1/*.jpg, data/person2/*.jpg, ...

        Incluir variaciones: iluminación, expresión, gafas, ángulos (si es posible).

        Detección y alineación de rostros

        Usar detector (Haar Cascade de OpenCV o dlib HOG/CNN) para recortar la región facial.

        Alinear ojos/nariz (si usan dlib con landmarks) para reducir variabilidad por rotación/pose.

        Resultado: imágenes recortadas donde el rostro ocupa la mayor parte y están alineadas.

    Preprocesamiento

        Convertir a escala de grises.

        Redimensionar todas a un mismo tamaño (ej. 112×92, 64×64 — tradeoff detalle/velocidad).

        Normalizar intensidad: restar media por imagen o dividir por std; también iluminación: histogram equalization / CLAHE si hace falta.

        (Opcional) Aplicar máscara para eliminar fondo irrelevante.

        Vectorizar imágenes

        Cada imagen m×n → vector columna de tamaño D = m*n.

        Construir matriz de datos X de tamaño D × N (N = número total de imágenes). Cada columna = imagen-vector.

        Cálculo de PCA (eigenfaces)

        Calcular la media μ (vector promedio) y centrar: X_c = X - μ.

        Es más eficiente usar SVD sobre la matriz de imágenes centradas o usar la técnica de la matriz de covarianza pequeña:

        Si D es grande y N < D, computar L = X_c^T X_c (N×N), obtener autovectores v_i de L, luego u_i = X_c v_i / sqrt(λ_i) → autovectores en dimensión D (eigenfaces).

        Ordenar componentes por autovalor (varianza explicada).

        Elegir k componentes (suma acumulada de varianza: ej. 90–95% o un número fijo como 50–150).

    Proyección

        Proyectar las imágenes centradas al subespacio PCA: y = U_k^T (x - μ) donde U_k contiene los k eigenfaces.

        Guardar las proyecciones (vectores de características) y etiquetas.

    Clasificación / Reconocimiento

        Método simple: Nearest Neighbor en espacio PCA (distancia euclidiana o coseno).

        Alternativas: k-NN, SVM con kernel lineal/RBF, discriminante lineal (LDA) sobre características PCA.

        Umbral de aceptación para ver si la cara pertenece a la base de datos (evitar falsos positivos).

    Evaluación (opcional)

        Separar datos en entrenamiento / prueba (por persona, por imagen; ej. leave-one-out, k-fold).

        Métricas: accuracy, precision/recall, confusion matrix, ROC/DET si discriminación clave.

        Medir tiempos (inferencia) y memoria.

        Demo en tiempo real (opcional)

        Capturar frame, detectar rostro, alinear, recortar, preprocesar, proyectar, buscar vecino más cercano y mostrar nombre / probabilidad.