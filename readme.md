# Detección de Spam con Machine Learning y Deep Learning

## Visión General

Este repositorio contiene una comparación exhaustiva de diferentes enfoques para la detección de spam en emails, incluyendo algoritmos tradicionales de machine learning, arquitecturas de deep learning y modelos avanzados de lenguaje (LLMs) como BERT.

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

4. **Enfoque con Modelos de Lenguaje (LLM)** ([`spam_con_LLM.ipynb`](spam_con_LLM.ipynb)):
   - Implementación con BERT (bert-base-uncased)
   - Fine-tuning para clasificación binaria
   - Evaluación comparativa

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

### Hallazgos Clave

**Machine Learning Tradicional:**
- **Mejor modelo:** SVM optimizado con kernel RBF
  - Accuracy: 97.38%
  - F1-score: 0.98
  - AUC-ROC: 0.996
- **Otros modelos destacados:**
  - Regresión Logística (Accuracy: 96.47%)
  - Bosque Aleatorio (Accuracy: 96.30%)

**Deep Learning:**
- **Mejor modelo:** MLP (Multilayer Perceptron)
  - Accuracy: 96.36%
  - Recall: 99%
  - F1-score: 0.97
  - AUC-ROC: 0.993
- **Comparativa de arquitecturas:**
  - LSTM: Accuracy 96.16%
  - RNN: Accuracy 94.79%

**Modelos de Lenguaje (LLM):**
- **BERT:**
  - Accuracy: 98.20%
  - Precision: 0.98
  - Recall: 0.99
  - F1-score: 0.98
  - AUC-ROC: 1.00

**Comparativa general:**
1. BERT (LLM) - Mejor rendimiento global pero mayor coste computacional
2. SVM (ML) - Excelente equilibrio rendimiento/eficiencia
3. MLP (DL) - Buen rendimiento, especialmente en recall

Las métricas de rendimiento completas y visualizaciones (matrices de confusión, curvas ROC) para todos los modelos están disponibles en la carpeta [resultados](resultados/), organizadas por enfoque (ML/DL/LLM)

## Cómo Usar

### Prerrequisitos
- Python 3.8+
- Paquetes requeridos (ver [`requerimientos.txt`](requerimientos.txt)):
  - pandas, scikit-learn, torch, transformers
  - tensorflow/keras, nltk, matplotlib
  - optuna (para optimización de hiperparámetros)

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

- Por último, spam_con_LLM.ipynb para enfoques de LLM

## Estructura del Proyecto
```
TFM/
├── datasets/                  # Archivos de datasets crudos
│   ├── enron_spam_data.csv
│   ├── huggingface_test.csv
│   ├── huggingface_train.csv
│   ├── spam.csv
│   └── spamassassin.csv
├── resultados/                # Visualizaciones de resultados
│   ├── DL/                    # Resultados de deep learning
│   ├── LLM/                   # Resultados con BERT
│   └── ML/                    # Resultados de machine learning
├── spam_con_DL.ipynb          # Implementación de deep learning
├── spam_con_LLM.ipynb         # Implementación con BERT
├── spam_con_ML.ipynb          # Implementación de machine learning
├── tratarDatos.py             # Módulo de preprocesamiento
├── README.md                  # Este archivo
└── requerimientos.txt         # Dependencias
```

## Contribuciones y Feedback

¡Las contribuciones son bienvenidas! Siéntete libre de:
- Reportar issues para errores o sugerencias
- Hacer fork del repositorio y crear pull requests
- Compartir tus resultados con diferentes configuraciones de parámetros

Para preguntas o feedback, por favor contacta a [salberichv@uoc.edu](mailto:salberichv@uoc.edu?subject=[GitHub]%20TFM%20repo%20doubts).