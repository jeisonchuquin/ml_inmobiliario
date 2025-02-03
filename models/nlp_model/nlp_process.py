import pandas as pd
import spacy
import nlp
import re

def extraer_numeros(texto: str) -> float:
    """
        Verifica si el valor es None o float
        Si el valor es un string, proceder con la limpieza
        Reemplaza separador de miles (coma o punto) por nada
        Reemplaza el último separador decimal con un punto
    """
    
    if isinstance(texto, float) or texto is None:
        return texto  # Dejar los floats como están, o retornar None para valores nulos

    texto = str(texto).replace('m2', '').strip()  # Eliminar 'm2' si está presente
    # Buscar número en el formato especificado
    match = re.search(r'-?\d{1,3}(?:[.,]\d{3})*(?:[.,]\d+)?', texto)
    if match:
        numero = match.group(0)
        numero = re.sub(r'(?<=\d)[,.](?=\d{3}(?:$|[.,]))', '', numero)
        numero = numero.replace(',', '.') if ',' in numero else numero
        return float(numero)
    
    return None

def detectar_atipicos_iqr(column):
    """
        Detecta valores atípicos usando el rango intercuartílico (IQR)
    """
    
    Q1 = column.quantile(0.25)
    Q3 = column.quantile(0.75)
    IQR = Q3 - Q1
    atipicos = (column < (Q1 - 1.5 * IQR)) | (column > (Q3 + 1.5 * IQR))
    return atipicos

def limpiar_texto(texto: str):
    """
        Limpia texto y elimina stopwords
        Elimina caracteres especiales
        Procesa el texto con spaCy y eliminar stopwords
    """
    
    texto = texto.replace('ñ', 'n')
    texto = re.sub(r'[^a-zA-Z0-9\s]', '', texto)
    doc = nlp(texto.lower())
    palabras = [token.text for token in doc if not token.is_stop]
    
    return ' '.join(palabras)

def cambiar_tipo_dato(data: pd.DataFrame)-> pd.DataFrame:
    """
        Cambia las variable al formato adecuado
    
    """
    
    df = data.copy()
    df['LIMITES'] = df['LIMITES'].astype(str)
    df['AREAPROPIEDAD'] = df['AREAPROPIEDAD'].astype(str)
    df['AREACONSTRUCCION'] = df['AREACONSTRUCCION'].astype(str)
    df['AREATERRENO'] = df['AREATERRENO'].astype(str)
    
    return df

def cargar_datos(ruta_archivo: str) -> pd.DataFrame:
    """
    Carga los datos desde un archivo CSV.

    params:
    ruta_archivo: Ruta del archivo CSV.
    return: DataFrame con los datos cargados.
    """
    
    return pd.read_csv(ruta_archivo, encoding='latin1')

def eliminar_nan(data: pd.DataFrame, columna: str) -> pd.DataFrame:
    """
        Elimina los valores NA de la columna ingresada

        parmas:
        data: DataFrame de pandas
        columna: Nombre de la columna de la que se eliminarán los valores NA
        return: DataFrame con los valores NA eliminados en la columna especificada
    """
    
    return data.dropna(subset=[columna])

def convertir_a_float(df, columna) -> pd.Series:
    """
    Convierte los valores de una columna a tipo float.

    params:
    data: DataFrame de pandas.
    columna: Nombre de la columna que se va a convertir.
    return: Serie de pandas con los valores convertidos a float.
    """
    
    df[columna] = df[columna].apply(extraer_numeros)
    
    return pd.to_numeric(df[columna], errors='coerce')

def procesar_datos(ruta_archivo: str) -> pd.DataFrame:
    """
    Procesa los datos del archivo CSV y los prepara para el análisis.

    params:
    ruta_archivo: Ruta del archivo CSV.
    return: DataFrame procesado.
    """
    
    df = cargar_datos(ruta_archivo)
    df = eliminar_nan(df, 'AVALUOCALCULADO')
    df = cambiar_tipo_dato(df)
    
    df['AREA_PROPIEDAD'] = convertir_a_float(df, 'AREAPROPIEDAD')
    df['AREA_CONSTRUCCION'] = convertir_a_float(df, 'AREACONSTRUCCION')
    df['AREA_TERRENO'] = convertir_a_float(df, 'AREATERRENO')
    
    df['AREA_PROPIEDAD'] = df['AREA_PROPIEDAD'].combine_first(df['AREA_TERRENO'])
    columnas_texto = ['LOCALIZACION', 'TIPO', 'SENALAMIENTO', 'LOCALIZACIONDELBIEN',
                    'DEPENDENCIAJURISDICCIONAL', 'SECTOR', 'UBICACION', 'LIMITES', 'CARACTERISTICAS']
    
    for columna in columnas_texto:
        df[columna] = df[columna].astype(str)
    
    atipicos_area_propiedad = detectar_atipicos_iqr(df['AREA_PROPIEDAD'])
    atipicos_area_construccion = detectar_atipicos_iqr(df['AREA_CONSTRUCCION'])
    atipicos_avaluo = detectar_atipicos_iqr(df['AVALUOCALCULADO'])
    
    df = df[~atipicos_area_propiedad & ~atipicos_area_construccion & ~atipicos_avaluo]
    
    nlp = spacy.load('es_core_news_sm')
    
    df['text_features'] = df[['LOCALIZACIONDELBIEN', 'DEPENDENCIAJURISDICCIONAL', 'SECTOR',
                            'UBICACION', 'LIMITES', 'CARACTERISTICAS']].agg(' '.join, axis=1)
    df['text_features'] = df['text_features'].apply(limpiar_texto)
    
    df['AREA_PROPIEDAD'] = df['AREA_PROPIEDAD'].fillna(df['AREA_PROPIEDAD'].mean())
    df['AREA_CONSTRUCCION'] = df['AREA_CONSTRUCCION'].fillna(df['AREA_CONSTRUCCION'].mean())
    df['AREA_CONSTRUCCION'] = df['AREA_CONSTRUCCION'].round(2)
    
    data = df[['CODIGO', 'LOCALIZACION', 'TIPO', 'SENALAMIENTO', 'text_features',
            'AREA_PROPIEDAD', 'AREA_CONSTRUCCION', 'AVALUOCALCULADO']]
    
    return data


ruta_archivo = r'data\raw\data_inmobiliario.csv'
data = procesar_datos(ruta_archivo)

