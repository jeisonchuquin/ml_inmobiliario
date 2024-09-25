import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import spacy
from transformers import pipeline

df = pd.read_csv(r'data\raw\data_inmobiliario.csv', encoding='latin1')### utf-16
df.columns




# Crear un DataFrame de ejemplo con una columna de texto
data = {'text': ["Apple is looking at buying U.K. startup for $1 billion to expand its international presence.",
                "Elon Musk founded SpaceX in 2002 with the goal of reducing space transportation costs and enabling the colonization of Mars.",
                "Google was founded in September 1998 by Larry Page and Sergey Brin, two PhD students at Stanford University."]}
df = pd.DataFrame(data)

# Cargar un modelo de resumen
summarizer = pipeline("summarization", framework="pt")

# Función para resumir el texto
def summarize_text(text):
    summary = summarizer(text, max_length=80, min_length=1, do_sample=False)
    return summary[0]['summary_text']

# Reemplazar los valores NaN con una cadena vacía
df['CARACTERISTICAS'] = df['CARACTERISTICAS'].fillna("")
df_copy = df.copy()

df= df.head(25)
# Aplicar la función de resumen a la columna
df['summary'] = df['CARACTERISTICAS'].apply(summarize_text)

# Mostrar el DataFrame con la nueva columna
print(df)


import re


# Función para extraer el número de habitaciones o cuartos
def extract_rooms(text):
    match = re.search(r"(habitaciones|cuartos):\s*(\d+)", text, re.IGNORECASE)
    return int(match.group(2)) if match else None


# Función para extraer el número de baños
def extract_bathrooms(text):
    match = re.search(r"Baños:\s*(\d+)", text)
    return int(match.group(1)) if match else None

# Crear nuevas columnas aplicando las funciones de extracción
df['Habitaciones'] = df['CARACTERISTICAS'].apply(extract_rooms)
df['Baños'] = df['CARACTERISTICAS'].apply(extract_bathrooms)

####prueba con bert
from transformers import BertTokenizer, BertForSequenceClassification
from preprocessing import preprocessing_pipeline
import torch

from sklearn.preprocessing import StandardScaler