import pandas as pd
import re
from string import punctuation
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split

def cargar_datos_huggingface():
    """Cargar y procesar el conjunto de datos de Hugging Face."""
    df_hugging1 = pd.read_csv("datasets/huggingface_train.csv")
    df_hugging2 = pd.read_csv("datasets/huggingface_test.csv")
    df_hugging = pd.concat([df_hugging1, df_hugging2])
    df_hugging = df_hugging.rename(columns={'text': 'email'})
    df_hugging['target'] = df_hugging['label'].map({'spam': 1, 'not_spam': 0})
    return df_hugging.drop(columns=['label'])

def cargar_datos_spam():
    """Cargar y procesar el conjunto de datos de spam."""
    df_spam = pd.read_csv("datasets/spam.csv")
    df_spam = df_spam.rename(columns={'v2': 'email'})
    df_spam['target'] = df_spam['v1'].map({'spam': 1, 'ham': 0})
    return df_spam.drop(columns=['v1', 'Unnamed: 0'])

def cargar_datos_spamassassin():
    """Cargar y procesar el conjunto de datos de SpamAssassin."""
    df_assassin = pd.read_csv("datasets/spamassassin.csv")
    df_assassin = df_assassin.rename(columns={'text': 'email'})
    df_assassin['target'] = df_assassin['label'].map({0: 1, 1: 0})
    return df_assassin.drop(columns=['label', 'group'])

def cargar_datos_enron():
    """Cargar y procesar el conjunto de datos de Enron."""
    df_enron = pd.read_csv("datasets/enron_spam_data.csv")
    df_enron = df_enron[['Message', 'Spam/Ham']].rename(columns={'Message': 'email'})
    df_enron['target'] = df_enron['Spam/Ham'].map({'spam': 1, 'ham': 0})
    return df_enron.drop(columns=['Spam/Ham'])

def limpiar_regex(m):
    """Limpiar el texto del correo electrónico utilizando expresiones regulares."""
    m = re.sub(r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$', '', str(m)) # Eliminar direcciones de correo electrónico
    m = re.sub(r'(http|https|ftp)://[a-zA-Z0-9\\./]+', ' ', str(m)) # Eliminar URLs
    m = re.sub(r'\d+', '', str(m)) # Eliminar números
    m = re.sub(r'<[^<]+?>', '', str(m)) # Eliminar etiquetas HTML
    m = m.replace(r'[^a-zA-Z]', '') # Eliminar caracteres no alfanuméricos
    m = m.replace('nbsp', '') # Eliminar artefactos comunes en spam
    m = m.translate(str.maketrans('', '', punctuation)) # Eliminar puntuación
    m = m.lower() # Convertir a minúsculas
    return m

def limpiar_columna(df, col_name):
    """Limpiar la columna especificada en el DataFrame aplicando pasos de preprocesamiento de texto."""
    stop_words = set(stopwords.words('english'))
    lem = WordNetLemmatizer()
    
    # Aplicar limpieza con regex
    df[col_name] = df[col_name].apply(limpiar_regex)
    # Eliminar palabras vacías (stopwords)
    df[col_name] = df[col_name].apply(lambda x: ' '.join([item for item in x.split() if item not in stop_words]))
    # Mantener solo palabras entre 3 y 15 caracteres
    df[col_name] = df[col_name].apply(lambda x: ' '.join([item for item in x.split() if 3 <= len(item) <= 15]))
    # Aplicar lematización (verbos y sustantivos)
    df[col_name] = df[col_name].apply(lambda x: ' '.join([lem.lemmatize(word, pos='v') for word in x.split()]))
    df[col_name] = df[col_name].apply(lambda x: ' '.join([lem.lemmatize(word, pos='n') for word in x.split()]))
    
    return df

def combinar_y_dividir_datos():
    """Combinar todos los conjuntos de datos, limpiarlos y dividirlos en conjuntos de entrenamiento y prueba."""
    df_hugging = cargar_datos_huggingface()
    df_spam = cargar_datos_spam()
    df_assassin = cargar_datos_spamassassin()
    df_enron = cargar_datos_enron()
    
    # Combinar conjuntos de datos
    df_todos = pd.concat([df_hugging, df_enron, df_spam, df_assassin])
    df_todos = df_todos.sample(frac=1).reset_index(drop=True)  # Barajar datos
    df_todos = df_todos.dropna().drop_duplicates()  # Eliminar nulos y duplicados
    
    # Limpiar la columna de correos electrónicos
    df_todos = limpiar_columna(df_todos, 'email')
    
    # Dividir datos en conjuntos de entrenamiento y prueba
    xtrain, xtest, ytrain, ytest = train_test_split(df_todos['email'], df_todos['target'], test_size=0.3, random_state=5)
    return xtrain, xtest, ytrain, ytest

if __name__ == "__main__":
    xtrain, xtest, ytrain, ytest = combinar_y_dividir_datos()
    spam=ytrain.eq(1).sum() + ytest.eq(1).sum()
    ham=ytrain.eq(0).sum() + ytest.eq(0).sum()
    print(f'Tamaño del conjunto de entrenamiento: {len(xtrain)}, Tamaño del conjunto de prueba: {len(xtest)}')
    print(f'Hay {spam} correos electrónicos spam y {ham} correos electrónicos ham en el dataset')

