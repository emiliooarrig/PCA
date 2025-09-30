PROYECTO MEDIO TERMINO
ALGEBRA LINEAL AVANZADA

EMILIO GUZMAN
FERNANDO REVILLA

SISTEMA RECONOCIMIENTO FACIAS USANDO ANALISIS DE COMPONENTES PRINCIPALES

Objetivos:

    conseguir 5 fotos por "persona"

    procesar las fotos a informacion que pueda ser utilizada por el sistema (automaticamente?)

    transformacion de imagenes a matrices

    analisis de componentes y proyecciones

    clasificación de imagenes (pruebas de que funciona, automatizado?)





Pasos propuestos:

    Recolección de datos

        Reunir carpetas por identidad: data/person1/*.jpg, data/person2/*.jpg, ...

        Incluir variaciones: iluminación, expresión, gafas, ángulos (si es posible).

        Detección ,alineación y recorte de rostros

    Preprocesamiento

        Convertir a escala de grises.

        Redimensionar todas a un mismo tamaño (ej. 112×92, 64×64).

        Normalizar intensidad (Opcional) Aplicar máscara para eliminar fondo irrelevante.

        Vectorizar imágenes

        Cada imagen m×n → vector columna de tamaño D = m*n.

        Construir matriz de datos X de tamaño D × N (N = número total de imágenes). Cada columna = imagen-vector.

        Cálculo de PCA 

        Ordenar componentes por autovalor .

        Elegir k componentes .

    Proyección

        Proyectar las imágenes centradas al subespacio PCA
        Guardar las proyecciones (vectores de características) y etiquetas.

    Clasificación / Reconocimiento

        Método simple: Nearest Neighbor en espacio PCA (distancia euclidiana o coseno).

        Umbral de aceptación para ver si la cara pertenece a la base de datos (evitar falsos positivos).
