import pandas as pd
import spacy
import numpy as np
import re


df = pd.read_csv(r'data\raw\data_inmobiliario.csv', encoding='latin1')### utf-16
df = df.dropna(subset=['AVALUOCALCULADO'])
df['LIMITES'] = df['LIMITES'].astype(str)
df['AREAPROPIEDAD'] = df['AREAPROPIEDAD'].astype(str)
df['AREACONSTRUCCION'] = df['AREACONSTRUCCION'].astype(str)
df['AREATERRENO'] = df['AREATERRENO'].astype(str)


def extraer_numeros(texto):
    # Verificar si el valor es None o float
    if isinstance(texto, float) or texto is None:
        return texto  # Dejar los floats como están, o retornar None para valores nulos
    # Si el valor es un string, proceder con la limpieza
    texto = str(texto).replace('m2', '').strip()  # Eliminar 'm2' si está presente
    # Buscar número en el formato especificado
    match = re.search(r'-?\d{1,3}(?:[.,]\d{3})*(?:[.,]\d+)?', texto)
    if match:
        numero = match.group(0)
        # Reemplazar separador de miles (coma o punto) por nada
        numero = re.sub(r'(?<=\d)[,.](?=\d{3}(?:$|[.,]))', '', numero)
        # Reemplazar el último separador decimal con un punto
        numero = numero.replace(',', '.') if ',' in numero else numero
        #numero = numero.replace('.', ',')
        return float(numero)
    return None



# Aplicar la función y convertir a float
df['AREA_PROPIEDAD'] = df['AREAPROPIEDAD'].apply(extraer_numeros)
df['AREA_PROPIEDAD'] = pd.to_numeric(df['AREA_PROPIEDAD'], errors='coerce')
df['AREA_CONSTRUCCION'] = df['AREACONSTRUCCION'].apply(extraer_numeros)
df['AREA_CONSTRUCCION'] = pd.to_numeric(df['AREA_CONSTRUCCION'], errors='coerce')
df['AREA_TERRENO'] = df['AREATERRENO'].apply(extraer_numeros)
df['AREA_TERRENO'] = pd.to_numeric(df['AREA_TERRENO'], errors='coerce')

df['AREA_PROPIEDAD'] = df['AREA_PROPIEDAD'].combine_first(df['AREA_TERRENO'])
df['LOCALIZACION'] = df['LOCALIZACION'].astype(str)
df['TIPO'] = df['TIPO'].astype(str)
df['SENALAMIENTO'] = df['SENALAMIENTO'].astype(str)
df['LOCALIZACIONDELBIEN'] = df['LOCALIZACIONDELBIEN'].astype(str)
df['DEPENDENCIAJURISDICCIONAL'] = df['DEPENDENCIAJURISDICCIONAL'].astype(str)
df['SECTOR'] = df['SECTOR'].astype(str)
df['UBICACION'] = df['UBICACION'].astype(str)
df['LIMITES'] = df['LIMITES'].astype(str)
df['CARACTERISTICAS'] = df['CARACTERISTICAS'].astype(str)

#Función para detectar valores atípicos usando el rango intercuartílico (IQR)
def detectar_atipicos_iqr(column):
    Q1 = column.quantile(0.25)
    Q3 = column.quantile(0.75)
    IQR = Q3 - Q1
    atipicos = (column < (Q1 - 1.5 * IQR)) | (column > (Q3 + 1.5 * IQR))
    return atipicos

# Detectar valores atípicos en AREA_PROPIEDAD y AREA_CONSTRUCCION
atipicos_area_propiedad = detectar_atipicos_iqr(df['AREA_PROPIEDAD'])
atipicos_area_construccion = detectar_atipicos_iqr(df['AREA_CONSTRUCCION'])
atipicos_avaluo =  detectar_atipicos_iqr(df['AREA_CONSTRUCCION'])
# Filtrar filas que no tienen valores atípicos
df = df[~atipicos_area_propiedad & ~atipicos_area_construccion & ~atipicos_avaluo]

nlp = spacy.load('es_core_news_sm')
# Función para limpiar texto y eliminar stopwords
def limpiar_texto(texto):
    # Eliminar caracteres especiales
    texto = texto.replace('ñ', 'n')
    texto = re.sub(r'[^a-zA-Z0-9\s]', '', texto)
    # Procesar el texto con spaCy y eliminar stopwords
    doc = nlp(texto.lower())
    palabras = [token.text for token in doc if not token.is_stop]
    return ' '.join(palabras)


df['text_features'] = df[['LOCALIZACIONDELBIEN',
'DEPENDENCIAJURISDICCIONAL','SECTOR','UBICACION','LIMITES','CARACTERISTICAS']].agg(' '.join, axis=1)
df['text_features'] = df['text_features'].apply(limpiar_texto)
df['AREA_PROPIEDAD'] = df['AREA_PROPIEDAD'].fillna(df['AREA_PROPIEDAD'].mean())
df['AREA_CONSTRUCCION'] = df['AREA_CONSTRUCCION'].fillna(df['AREA_CONSTRUCCION'].mean())
df['AREA_CONSTRUCCION'] = df['AREA_CONSTRUCCION'].round(2)
df_f = df[['CODIGO', 'LOCALIZACION', 'TIPO', 'SENALAMIENTO', 'text_features','AREA_PROPIEDAD', 'AREA_CONSTRUCCION','AVALUOCALCULADO']]




