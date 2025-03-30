# Detección de Spam con Machine Learning y Deep Learning

## Visión General

Este repositorio contiene una comparación exhaustiva de diferentes enfoques para la detección de spam en emails, que van desde algoritmos tradicionales de machine learning hasta arquitecturas avanzadas de deep learning.

### Componentes Clave:

1. **Módulo de Preprocesamiento** ([``tratarDatos.py``](tratarDatos.py)):
   - Maneja la carga y combinación de múltiples conjuntos de datos de spam
   - Implementa limpieza de texto (regex, eliminación de stopwords, lematización)
   - Realiza división train-test

2. **Enfoque de Machine Learning** ([``spam_con_ML.ipynb``](spam_con_ML.ipynb)):
   - Implementa 8 algoritmos ML tradicionales
   - Incluye optimización de hiperparámetros

3. **Enfoque de Deep Learning** ([``spam_con_DL.ipynb``](spam_con_DL.ipynb)):
   - Implementa 3 arquitecturas neuronales (LSTM, RNN, MLP)
   - Optimización con Optuna

## Conjuntos de Datos

Se utilizan cuatro conjuntos de datos públicos combinados:
1. **Enron Spam Dataset:** Contiene emails de empleados de Enron
2. **SpamAssassin Dataset:** Colección pública de mensajes spam y ham
3. **HuggingFace Spam Dataset:** Colección moderna de spam de HuggingFace
4. **SMS Spam Collection:** Clásico conjunto de datos de SMS spam

El dataset combinado contiene:
- Total de emails: 51709
- Proporción de spam: 42,38% (21916)
- Proporción de ham: 57,62% (29793)
## Resultados y Análisis

Las métricas de rendimiento (exactitud, precisión, recall, F1-score) y visualizaciones (matrices de confusión, curvas ROC, gráficos de pérdida) para todos los modelos están disponibles en la carpeta [resultados](resultados/), organizadas por enfoque (ML/DL).

Hallazgos clave (FALTA RELLENAR):
- Mejor modelo ML: [Nombre del Modelo] con [Exactitud]% de precisión
- Mejor modelo DL: [Nombre del Modelo] con [Exactitud]% de precisión
- Comparación de requisitos computacionales y tiempo de entrenamiento

## Cómo Usar

### Prerrequisitos
- Python 3.8+
- Paquetes requeridos:
  - pandas
  - scikit-learn
  - torch
  - tensorflow/keras
  - nltk
  - matplotlib

### Instalación
1. Clona el repositorio:
```
   git clone https://github.com/sergial273/TFM
```

2. Instala las dependencias:
```
   pip install -r requerimientos.txt
```

3. Descarga los datos de NLTK:
```
   import nltk
   nltk.download('stopwords')
   nltk.download('wordnet')
```

### Ejecución del Código
- Primero spam_con_ML.ipynb para enfoques de machine learning

- Luego spam_con_DL.ipynb para enfoques de deep learning

## Estructura del Proyecto
```
TFM/
├── datasets/                  # Archivos de datasets crudos
├── resultados/                # Visualizaciones de resultados
│   ├── DL/                    # Resultados de deep learning
│   └── ML/                    # Resultados de machine learning
├── spam_con_DL.ipynb          # Implementación de deep learning
├── spam_con_ML.ipynb          # Implementación de machine learning
└── tratarDatos.py             # Módulo de preprocesamiento
```

## Contribuciones y Feedback

¡Las contribuciones son bienvenidas! Siéntete libre de:
- Reportar issues para errores o sugerencias
- Hacer fork del repositorio y crear pull requests
- Compartir tus resultados con diferentes configuraciones de parámetros

Para preguntas o feedback, por favor contacta a [salberichv@uoc.edu](mailto:salberichv@uoc.edu?subject=[GitHub]%20TFM%20repo%20doubts).