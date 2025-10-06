# Proyecto de Medio Término — Álgebra Lineal Avanzada

## Integrantes  
- **Emilio Guzmán**  
- **Fernando Revilla**

---

## Sistema de Reconocimiento Facial usando Análisis de Componentes Principales (PCA)

### Objetivos
- Obtener **5 fotografías de una persona** para conformar la base de datos.  
- **Procesar las imágenes automáticamente** para generar datos útiles para el sistema.  
- **Transformar las imágenes en matrices numéricas** que representen su información visual.  
- Aplicar **Análisis de Componentes Principales (PCA)** para reducir la dimensionalidad y extraer características relevantes.  
- Implementar un sistema de **clasificación y reconocimiento facial**, verificando su correcto funcionamiento con pruebas automatizadas.

---

## Metodología y Pasos Propuestos

### 1. Recolección de Datos  
- Crear carpetas por identidad:  - Incluyendo variaciones en las expresiones faciales en las imágenes:  

---

### 2. Preprocesamiento de Imágenes  
- **Convertir a escala de grises** para simplificar el análisis. Agregandolas a un carpeta con el nombre de `normalized_data`.  
- **Redimensionar todas las imágenes** a un mismo tamaño (para nuestro sistema usaremos `60x90` ).  
- **Vectorizar cada imagen:**  
- Una imagen de tamaño `m×n` se convierte en un vector columna de tamaño `D = m * n`.  
- **Construir la matriz de datos X:**  
- Tamaño: `D × N`, donde `N` es el número total de imágenes.  
- Cada fila representa una imagen vectorizada.  

---

### 3. Cálculo de PCA  
- Calcular la **matriz de covarianza** y obtener **autovalores y autovectores**.  
- Ordenar los componentes principales por su **autovalor** (importancia).  
- Seleccionar los **k componentes principales** que conserven la mayor varianza.

---

### 4. Proyección  
- **Centrar las imágenes** y proyectarlas en el **subespacio PCA**.  
- Guardar los **vectores de características** (proyecciones) junto con sus **etiquetas de identidad**.

---

### 5. Clasificación y Reconocimiento  
- Implementar un método simple de reconocimiento: **Nearest Neighbor (Vecino más cercano)**.  
- Establecer un **umbral de aceptación** para determinar si una cara pertenece o no a la base de datos, reduciendo falsos positivos.

---

## Resultados Esperados
- Identificación correcta de rostros previamente registrados.  
- Visualización de los **autovectores principales (eigenfaces)**.  
- Gráficas de **varianza explicada** para seleccionar el número óptimo de componentes `k`.  

---




