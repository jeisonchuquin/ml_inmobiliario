import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import spacy
from transformers import pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.impute import SimpleImputer
import numpy as np


import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(r'data\raw\data_inmobiliario.csv', encoding='latin1')### utf-16
df = df.dropna(subset=['AVALUOCALCULADO'])
df['LIMITES'] = df['LIMITES'].astype(str)
df['AREAPROPIEDAD'] = df['AREAPROPIEDAD'].astype(str)
df['AREACONSTRUCCION'] = df['AREACONSTRUCCION'].astype(str)
df['AREATERRENO'] = df['AREATERRENO'].astype(str)

df.columns
df.head(5)

# Cargar el modelo en español de spaCy
nlp = spacy.load('es_core_news_sm')
def extraer_area_principal_spacy(texto):
    doc = nlp(texto)
    area_principal = None
    
    # Recorrer los tokens y buscar cantidades seguidas de 'm2'
    for i, token in enumerate(doc):
        if token.like_num:  # Si es un número
            if i + 1 < len(doc) and doc[i + 1].text == 'm2':  # Verificar si hay un token siguiente
                area_principal = token.text
                break
    return area_principal



# Aplicar la función a la columna 'AREAPROPIEDAD'
df['AREA_PROPIEDAD'] = df['AREAPROPIEDAD'].apply(extraer_area_principal_spacy)
df['AREA_CONSTRUCCION'] = df['AREACONSTRUCCION'].apply(extraer_area_principal_spacy)
df['AREA_TERRENO'] = df['AREATERRENO'].apply(extraer_area_principal_spacy)

def clean_area(area_str):
    if isinstance(area_str, str):
        # Eliminar 'm2' y otros caracteres innecesarios, si existen
        area_str = area_str.replace('m2', '').strip()
        # Eliminar puntos que no son decimales
        area_str = area_str.replace('.', '')  # Eliminar miles
        # Reemplazar la última coma por un punto
        if ',' in area_str:
            area_str = area_str.rsplit(',', 1)
            area_str = '.'.join(area_str)
        return area_str
    return area_str  # Devolver tal cual si no es una cadena

# Aplicar la función y convertir a float
df['AREA_PROPIEDAD'] = (df['AREA_PROPIEDAD'].apply(clean_area).replace('', None))
df['AREA_PROPIEDAD'] = pd.to_numeric(df['AREA_PROPIEDAD'], errors='coerce')
df['AREA_CONSTRUCCION'] = (df['AREA_CONSTRUCCION'].apply(clean_area).replace('', None))
df['AREA_CONSTRUCCION'] = pd.to_numeric(df['AREA_CONSTRUCCION'], errors='coerce')
df['AREA_TERRENO'] = (df['AREA_TERRENO'].apply(clean_area).replace('', None))
df['AREA_TERRENO'] = pd.to_numeric(df['AREA_TERRENO'], errors='coerce')


# # Cargar un modelo de resumen
# summarizer = pipeline("summarization", framework="pt")

# # Función para resumir el texto
# def summarize_text1(text):
#     summary = summarizer(text, max_length=50, min_length=1, do_sample=False)
#     return summary[0]['summary_text']
# def summarize_text2(text):
#     summary = summarizer(text, max_length=30, min_length=1, do_sample=False)
#     return summary[0]['summary_text']
# def summarize_text3(text):
#     summary = summarizer(text, max_length=40, min_length=1, do_sample=False)
#     return summary[0]['summary_text']
# def summarize_text4(text):
#     summary = summarizer(text, max_length=100, min_length=1, do_sample=False)
#     return summary[0]['summary_text']



# df['DEPENDENCIAJURISDICCIONAL_'] = df['DEPENDENCIAJURISDICCIONAL'].fillna("").apply(summarize_text1)
# df['SECTOR_'] = df['SECTOR'].fillna("").apply(summarize_text2)
# df['UBICACION_'] = df['UBICACION'].fillna("").apply(summarize_text3)
# df['LIMITES_'] = df['LIMITES'].fillna("").apply(summarize_text3)
# df['CARACTERISTICAS_'] = df['CARACTERISTICAS'].fillna("").apply(summarize_text4)


#df.to_csv(r'data\raw\data_inmobiliario_summarize.csv', index=False)

df.columns

# Convertir las columnas a cadena para evitar errores de concatenación
df['LOCALIZACION'] = df['LOCALIZACION'].astype(str)
df['TIPO'] = df['TIPO'].astype(str)
df['SENALAMIENTO'] = df['SENALAMIENTO'].astype(str)
df['LOCALIZACIONDELBIEN'] = df['LOCALIZACIONDELBIEN'].astype(str)
# df['DEPENDENCIAJURISDICCIONAL_'] = df['DEPENDENCIAJURISDICCIONAL_'].astype(str)
# df['SECTOR_'] = df['SECTOR_'].astype(str)
# df['UBICACION_'] = df['UBICACION_'].astype(str)
# df['LIMITES_'] = df['LIMITES_'].astype(str)
# df['CARACTERISTICAS_'] = df['CARACTERISTICAS_'].astype(str)

df['DEPENDENCIAJURISDICCIONAL'] = df['DEPENDENCIAJURISDICCIONAL'].astype(str)
df['SECTOR'] = df['SECTOR'].astype(str)
df['UBICACION'] = df['UBICACION'].astype(str)
df['LIMITES'] = df['LIMITES'].astype(str)
df['CARACTERISTICAS'] = df['CARACTERISTICAS'].astype(str)


df_copy = df.copy()




# Visualización de las distribuciones con boxplots
plt.figure(figsize=(12, 6))

# Boxplot para AREA_PROPIEDAD
plt.subplot(1, 2, 1)
sns.boxplot(x=df['AREA_PROPIEDAD'])
plt.title('Boxplot de AREA_PROPIEDAD')

# Boxplot para AREA_CONSTRUCCION
plt.subplot(1, 2, 2)
sns.boxplot(x=df['AREA_CONSTRUCCION'])
plt.title('Boxplot de AREA_CONSTRUCCION')

plt.show()

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
# Mostrar cuántos valores atípicos hay en cada columna
print(f"Valores atípicos en AREA_PROPIEDAD: {atipicos_area_propiedad.sum()}")
print(f"Valores atípicos en AREA_CONSTRUCCION: {atipicos_area_construccion.sum()}")
print(f"Valores atípicos en AREA_CONSTRUCCION: {atipicos_avaluo.sum()}")
# Filtrar filas que no tienen valores atípicos
df = df[~atipicos_area_propiedad & ~atipicos_area_construccion & ~atipicos_avaluo]

print(f"Datos después de eliminar valores atípicos: {df.shape}")


df = df.head(200)







# Asegúrate de que eliminas los valores en todas las columnas relevantes
df_sin_atipicos = df[~atipicos_area_propiedad & ~atipicos_area_construccion & ~atipicos_avaluo]

df_sin_atipicos['text_features'] = df_sin_atipicos[['LOCALIZACION', 'TIPO', 'SENALAMIENTO','LOCALIZACIONDELBIEN',
'DEPENDENCIAJURISDICCIONAL','SECTOR','UBICACION','LIMITES','CARACTERISTICAS']].agg(' '.join, axis=1)



# Verificar el tamaño después de eliminar atípicos
print(f"Datos después de eliminar valores atípicos: {df_sin_atipicos.shape}")

# Separar características textuales y numéricas
X_text = df_sin_atipicos[['text_features']]  # Características textuales
X_numeric = df_sin_atipicos[['AREA_PROPIEDAD', 'AREA_CONSTRUCCION']]  # Características numéricas
y = df_sin_atipicos['AVALUOCALCULADO']  # Variable objetivo
X_text.shape, X_numeric.shape, y.shape
# Combinar características textuales y numéricas en un solo DataFrame
X_combined = pd.concat([X_text, X_numeric], axis=1)

# Verificar tamaños antes de dividir
print(f"Tamaño de X_combined: {X_combined.shape}")
print(f"Tamaño de y: {y.shape}")

# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42)


# # Concatenar las columnas de texto en una sola columna
# df['text_features'] = df[['LOCALIZACION', 'TIPO', 'SENALAMIENTO','LOCALIZACIONDELBIEN',
# 'DEPENDENCIAJURISDICCIONAL_','SECTOR_','UBICACION_','LIMITES_','CARACTERISTICAS_']].agg(' '.join, axis=1)

import re
from nltk.corpus import stopwords
import nltk

# Función para limpiar texto y eliminar stopwords
def limpiar_texto(texto):
    # Eliminar caracteres especiales
    texto = texto.replace('ñ', 'n')
    texto = re.sub(r'[^a-zA-Z0-9\s]', '', texto)
    # Procesar el texto con spaCy y eliminar stopwords
    doc = nlp(texto.lower())
    palabras = [token.text for token in doc if not token.is_stop]
    return ' '.join(palabras)


df['text_features'] = df[['LOCALIZACION', 'TIPO', 'SENALAMIENTO','LOCALIZACIONDELBIEN',
'DEPENDENCIAJURISDICCIONAL','SECTOR','UBICACION','LIMITES','CARACTERISTICAS']].agg(' '.join, axis=1)
df['text_features'] = df['text_features'].apply(limpiar_texto)
df_f = df[['CODIGO', 'text_features','AREA_PROPIEDAD', 'AREA_CONSTRUCCION','AVALUOCALCULADO']]



# Separar características textuales y numéricas
X_text = df[['text_features']]  # Características textuales
X_numeric = df[['AREA_PROPIEDAD', 'AREA_CONSTRUCCION']]#, 'AREA_TERRENO']]  # Características numéricas
y = df['AVALUOCALCULADO']

# Combinar características textuales y numéricas en un solo DataFrame
X_combined = pd.concat([X_text, X_numeric], axis=1)#.reset_index(drop=True)], axis=1)

# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42)

# Imprimir las formas para depuración
print("X_train shape:", X_train.shape)  # Debe ser (n_samples, n_features)
print("y_train shape:", y_train.shape)  # Debe ser (n_samples,)

# Crear un ColumnTransformer para procesar las características textuales y numéricas
preprocessor = ColumnTransformer(
    transformers=[
        ('text', TfidfVectorizer(), 'text_features'),  # Transformar características textuales
        ('num', SimpleImputer(strategy='mean'), ['AREA_PROPIEDAD', 'AREA_CONSTRUCCION'])#, 'AREA_TERRENO'])  # Imputar valores faltantes para características numéricas
    ],
    remainder='drop'  # Eliminar columnas que no se procesan
)

# Crear un pipeline para el preprocesamiento y el modelo
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor())  # Puedes cambiar a otro modelo si lo deseas
])

# Entrenar el modelo
try:
    # Usar todas las características combinadas
    pipeline.fit(X_train, y_train)  
except ValueError as e:
    print("Error durante el entrenamiento:", e)

# Paso 5: Predecir
predicciones = pipeline.predict(X_test)

# Mostrar resultados
print(predicciones)

# Calcular MAE y RMSE
mae = mean_absolute_error(y_test, predicciones)
rmse = np.sqrt(mean_squared_error(y_test, predicciones))

# Mostrar los resultados
print("Mean Absolute Error (MAE):", mae)
print("Root Mean Squared Error (RMSE):", rmse)






from sklearn.model_selection import GridSearchCV


# Parámetros a ajustar con GridSearchCV
param_grid = {
    'model__n_estimators': [100, 200, 300],  # Número de árboles
    'model__max_depth': [10, 20, 30, None],  # Profundidad máxima del árbol
    'model__min_samples_split': [2, 5, 10],  # Mínimo de muestras para dividir un nodo
    'model__min_samples_leaf': [1, 2, 4],    # Mínimo de muestras en una hoja
}

# Crear el GridSearchCV
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1, verbose=2)

# Entrenar el modelo con GridSearchCV
try:
    # Usar todas las características combinadas
    grid_search.fit(X_train, y_train)  
except ValueError as e:
    print("Error durante el entrenamiento:", e)

# Obtener los mejores hiperparámetros
print("Mejores hiperparámetros encontrados:", grid_search.best_params_)

# Usar el mejor modelo para predecir
best_model = grid_search.best_estimator_
predicciones = best_model.predict(X_test)

# Calcular MAE y RMSE
mae = mean_absolute_error(y_test, predicciones)
rmse = np.sqrt(mean_squared_error(y_test, predicciones))

# Mostrar los resultados
print("Mean Absolute Error (MAE):", mae)
print("Root Mean Squared Error (RMSE):", rmse)



### ultimo con los cambios ugeridos
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Asegurarse de que las columnas categóricas sean de tipo cadena
for col in ['LOCALIZACION', 'TIPO', 'SENALAMIENTO', 'LOCALIZACIONDELBIEN', 'DEPENDENCIAJURISDICCIONAL_', 
            'SECTOR_', 'UBICACION_', 'LIMITES_', 'CARACTERISTICAS_', 'DEPENDENCIAJURISDICCIONAL',
            'SECTOR', 'UBICACION', 'LIMITES', 'CARACTERISTICAS']:
    df[col] = df[col].astype(str)

# Concatenar las columnas de texto en una sola columna
df['text_features'] = df[['LOCALIZACION', 'TIPO', 'SENALAMIENTO','LOCALIZACIONDELBIEN',
                            'DEPENDENCIAJURISDICCIONAL_','SECTOR_','UBICACION_','LIMITES_','CARACTERISTICAS_']].agg(' '.join, axis=1)

df['text_features'] = df[['LOCALIZACION', 'TIPO', 'SENALAMIENTO','LOCALIZACIONDELBIEN',
                            'DEPENDENCIAJURISDICCIONAL','SECTOR','UBICACION','LIMITES','CARACTERISTICAS']].agg(' '.join, axis=1)

# Separar características textuales y numéricas
X_text = df[['text_features']]  # Características textuales
X_numeric = df[['AREA_PROPIEDAD', 'AREA_CONSTRUCCION', 'AREA_TERRENO']]  # Características numéricas
y = df['AVALUOCALCULADO']

# Combinar características textuales y numéricas en un solo DataFrame
X_combined = pd.concat([X_text, X_numeric.reset_index(drop=True)], axis=1)

# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42)

# Crear un ColumnTransformer para procesar las características textuales y numéricas
preprocessor = ColumnTransformer(
    transformers=[
        ('text', TfidfVectorizer(), 'text_features'),  # Transformar características textuales
        ('num', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())  # Escalar características numéricas
        ]), ['AREA_PROPIEDAD', 'AREA_CONSTRUCCION', 'AREA_TERRENO'])  # Imputar y escalar características numéricas
    ]
)

# Crear un pipeline con el preprocesamiento y el modelo XGBoost
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', XGBRegressor())
])

# Definir el conjunto de hiperparámetros para XGBoost
param_grid = {
    'model__n_estimators': [100, 300],
    'model__max_depth': [3, 6, 9],
    'model__learning_rate': [0.01, 0.1],
    'model__subsample': [0.7, 1],
    'model__colsample_bytree': [0.7, 1],
    'model__gamma': [0, 0.1, 0.3],  # Regularización
    'model__reg_alpha': [0, 1],  # Regularización L1
    'model__reg_lambda': [1, 2],  # Regularización L2
    'model__min_child_weight': [1, 5]  # Control de la complejidad
}

# Realizar la búsqueda de hiperparámetros con validación cruzada
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Obtener los mejores hiperparámetros
print("Mejores hiperparámetros encontrados:", grid_search.best_params_)

# Predecir en el conjunto de prueba con los mejores hiperparámetros
best_model = grid_search.best_estimator_
predicciones = best_model.predict(X_test)

# Calcular MAE y RMSE
mae = mean_absolute_error(y_test, predicciones)
rmse = np.sqrt(mean_squared_error(y_test, predicciones))

# Mostrar los resultados
print("Mean Absolute Error (MAE):", mae)
print("Root Mean Squared Error (RMSE):", rmse)

### continur a correr desde aqui

# Comparar con otros modelos: LightGBM, GradientBoostingRegressor, etc.
from lightgbm import LGBMRegressor
from sklearn.ensemble import GradientBoostingRegressor

# Crear pipelines para LightGBM y GradientBoostingRegressor
pipeline_lgb = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', LGBMRegressor())
])

pipeline_gb = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', GradientBoostingRegressor())
])

# Definir conjuntos de hiperparámetros para LightGBM y GradientBoostingRegressor
param_grid_lgb = {
    'model__n_estimators': [100, 300],
    'model__learning_rate': [0.01, 0.1],
    'model__max_depth': [3, 6, 9]
}

param_grid_gb = {
    'model__n_estimators': [100, 300],
    'model__learning_rate': [0.01, 0.1],
    'model__max_depth': [3, 6, 9]
}

# Búsqueda de hiperparámetros para LightGBM
grid_search_lgb = GridSearchCV(pipeline_lgb, param_grid_lgb, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1, verbose=2)
grid_search_lgb.fit(X_train, y_train)
print("Mejores hiperparámetros para LightGBM:", grid_search_lgb.best_params_)

# Búsqueda de hiperparámetros para GradientBoostingRegressor
grid_search_gb = GridSearchCV(pipeline_gb, param_grid_gb, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1, verbose=2)
grid_search_gb.fit(X_train, y_train)
print("Mejores hiperparámetros para GradientBoostingRegressor:", grid_search_gb.best_params_)












#####intento 10000
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, RepeatedKFold
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
import xgboost as xgb
import lightgbm as lgb
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Separar características textuales y numéricas
X_text = df[['text_features']]  # Características textuales
X_numeric = df[['AREA_PROPIEDAD', 'AREA_CONSTRUCCION', 'AREA_TERRENO']]  # Características numéricas
y = df['AVALUOCALCULADO']

# Combinar características textuales y numéricas en un solo DataFrame
X_combined = pd.concat([X_text, X_numeric.reset_index(drop=True)], axis=1)

# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42)

# Crear un ColumnTransformer con escalado numérico, imputación y PCA
preprocessor = ColumnTransformer(
    transformers=[
        ('text', TfidfVectorizer(), 'text_features'),  # Transformar características textuales
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),  # Imputar valores faltantes
            ('scaler', StandardScaler()),  # Escalar características numéricas
            ('pca', PCA(n_components=2))   # Reducir dimensionalidad
        ]), ['AREA_PROPIEDAD', 'AREA_CONSTRUCCION', 'AREA_TERRENO'])  # Escalar características numéricas
    ],
    remainder='drop'  # Eliminar columnas que no se procesan
)

# Repeated K-Fold
cv = RepeatedKFold(n_splits=5, n_repeats=2, random_state=42)

# Hiperparámetros ajustados para XGBoost
param_grid_xgb = {
    'model__n_estimators': [100, 300, 500],
    'model__max_depth': [5, 7, 9],
    'model__learning_rate': [0.01, 0.05, 0.1],
    'model__subsample': [0.7, 0.8, 0.9],
    'model__colsample_bytree': [0.7, 0.8, 1]
}

# Modelo XGBoost
pipeline_xgb = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', xgb.XGBRegressor(objective='reg:squarederror'))
])

grid_xgb = GridSearchCV(pipeline_xgb, param_grid_xgb, cv=cv, scoring='neg_mean_absolute_error', n_jobs=-1, verbose=1)
grid_xgb.fit(X_train, y_train)
best_params_xgb = grid_xgb.best_params_
print("Mejores hiperparámetros para XGBoost:", best_params_xgb)

# Evaluar XGBoost
y_pred_xgb = grid_xgb.predict(X_test)
mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
print("XGBoost - Mean Absolute Error (MAE):", mae_xgb)
print("XGBoost - Root Mean Squared Error (RMSE):", rmse_xgb)


# Repetir lo mismo para LightGBM
param_grid_lgb = {
    'model__n_estimators': [100, 300, 500],
    'model__max_depth': [5, 7, 9],
    'model__learning_rate': [0.01, 0.05, 0.1],
    'model__subsample': [0.7, 0.8, 0.9],
    'model__colsample_bytree': [0.7, 0.8, 1]
}

pipeline_lgb = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', lgb.LGBMRegressor())
])

grid_lgb = GridSearchCV(pipeline_lgb, param_grid_lgb, cv=cv, scoring='neg_mean_absolute_error', n_jobs=-1, verbose=1)
grid_lgb.fit(X_train, y_train)
best_params_lgb = grid_lgb.best_params_
print("Mejores hiperparámetros para LightGBM:", best_params_lgb)

# Evaluar LightGBM
y_pred_lgb = grid_lgb.predict(X_test)
mae_lgb = mean_absolute_error(y_test, y_pred_lgb)
rmse_lgb = np.sqrt(mean_squared_error(y_test, y_pred_lgb))
print("LightGBM - Mean Absolute Error (MAE):", mae_lgb)
print("LightGBM - Root Mean Squared Error (RMSE):", rmse_lgb)

# Modelo GradientBoostingRegressor
param_grid_gb = {
    'model__n_estimators': [100, 300, 500],
    'model__max_depth': [5, 7, 9],
    'model__learning_rate': [0.01, 0.05, 0.1]
}

pipeline_gb = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', GradientBoostingRegressor())
])

grid_gb = GridSearchCV(pipeline_gb, param_grid_gb, cv=cv, scoring='neg_mean_absolute_error', n_jobs=-1, verbose=1)
grid_gb.fit(X_train, y_train)
best_params_gb = grid_gb.best_params_
print("Mejores hiperparámetros para GradientBoostingRegressor:", best_params_gb)

# Evaluar GradientBoostingRegressor
y_pred_gb = grid_gb.predict(X_test)
mae_gb = mean_absolute_error(y_test, y_pred_gb)
rmse_gb = np.sqrt(mean_squared_error(y_test, y_pred_gb))
print("GradientBoostingRegressor - Mean Absolute Error (MAE):", mae_gb)
print("GradientBoostingRegressor - Root Mean Squared Error (RMSE):", rmse_gb)








































































import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.impute import SimpleImputer
import numpy as np

# Supongamos que tienes un dataframe df
# df = pd.read_csv("tu_archivo.csv")

# Paso 1: Preprocesar los datos
df = df.dropna(subset=['AVALUOCALCULADO'])


df['LOCALIZACION'] = df['LOCALIZACION'].astype(str)
df['TIPO'] = df['TIPO'].astype(str)
df['SENALAMIENTO'] = df['SENALAMIENTO'].astype(str)
df['LOCALIZACIONDELBIEN'] = df['LOCALIZACIONDELBIEN'].astype(str)
df['CARACTERISTICAS'] = df['CARACTERISTICAS'].astype(str)
# Concatenar las columnas de texto en una sola columna
df['text_features'] = df[['LOCALIZACION', 'TIPO', 'SENALAMIENTO', 'LOCALIZACIONDELBIEN', 'CARACTERISTICAS']].agg(' '.join, axis=1)

# Separar características y el objetivo
X_text = df[['text_features']]  # Características textuales
X_numeric = df[['AREA_PROPIEDAD', 'AREA_CONSTRUCCION']]  # Características numéricas
y = df['AVALUOCALCULADO']

# Dividir los datos en entrenamiento y prueba
X_train_text, X_test_text, X_train_numeric, X_test_numeric, y_train, y_test = train_test_split(
    X_text, X_numeric, y, test_size=0.2, random_state=42)

# Imprimir las formas para depuración
print("X_train_text shape:", X_train_text.shape)  # Debe ser (n_samples, 1)
print("X_train_numeric shape:", X_train_numeric.shape)  # Debe ser (n_samples, 2)
print("y_train shape:", y_train.shape)  # Debe ser (n_samples,)

# Crear un ColumnTransformer para procesar las características textuales y numéricas
preprocessor = ColumnTransformer(
    transformers=[
        ('text', TfidfVectorizer(), 'text_features'),  # Transformar características textuales
        ('num', SimpleImputer(strategy='mean'), ['AREA_PROPIEDAD', 'AREA_CONSTRUCCION'])  # Imputar valores faltantes para características numéricas
    ],
    remainder='drop'  # Eliminar columnas que no se procesan
)

# Crear un pipeline para el preprocesamiento y el modelo
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor())  # Puedes cambiar a otro modelo si lo deseas
])

# Entrenar el modelo
try:
    # Usar las características textuales y numéricas
    pipeline.fit(X_train_text['text_features'], y_train)  
except ValueError as e:
    print("Error durante el entrenamiento:", e)

# Paso 5: Predecir
predicciones = pipeline.predict(X_test_text['text_features'])

# Mostrar resultados
print(predicciones)

# Calcular MAE y RMSE
mae = mean_absolute_error(y_test, predicciones)
rmse = np.sqrt(mean_squared_error(y_test, predicciones))

# Mostrar los resultados
print("Mean Absolute Error (MAE):", mae)
print("Root Mean Squared Error (RMSE):", rmse)








####Este si funciona pero no uso numerico

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Supongamos que tienes un dataframe df
# df = pd.read_csv("tu_archivo.csv")

# Paso 1: Preprocesar los datos
df = df.dropna(subset=['AVALUOCALCULADO'])

# Concatenar las columnas de texto en una sola columna
df['text_features'] = df[['LOCALIZACION', 'TIPO', 'SENALAMIENTO', 'LOCALIZACIONDELBIEN', 'AREAPROPIEDAD', 'AREACONSTRUCCION', 'CARACTERISTICAS']].agg(' '.join, axis=1)

# Separar características y el objetivo
X = df[['text_features']]  # Mantener X como un DataFrame
y = df['AVALUOCALCULADO']

# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Imprimir las formas para depuración
print("X_train shape:", X_train.shape)  # Debe ser (n_samples, 1)
print("y_train shape:", y_train.shape)  # Debe ser (n_samples,)

# Crear un pipeline para el preprocesamiento y el modelo
pipeline = Pipeline(steps=[
    ('tfidf', TfidfVectorizer()),  # Aplicar TfidfVectorizer a la columna de texto
    ('model', RandomForestRegressor())  # Puedes cambiar a otro modelo si lo deseas
])

# Entrenar el modelo
try:
    # Asegúrate de pasar solo la columna de texto
    pipeline.fit(X_train['text_features'], y_train)  
except ValueError as e:
    print("Error durante el entrenamiento:", e)

# Paso 5: Predecir
predicciones = pipeline.predict(X_test['text_features'])  # Usa X_test['text_features'] para predecir

# Mostrar resultados
print(predicciones)

# Calcular MAE y RMSE
mae = mean_absolute_error(y_test, predicciones)
rmse = np.sqrt(mean_squared_error(y_test, predicciones))

# Mostrar los resultados
print("Mean Absolute Error (MAE):", mae)
print("Root Mean Squared Error (RMSE):", rmse)



import matplotlib.pyplot as plt

plt.scatter(y_test, predicciones)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Valores Reales')
plt.ylabel('Predicciones')
plt.title('Predicciones vs Valores Reales')
plt.show()


errores = predicciones - y_test
plt.hist(errores, bins=30)
plt.xlabel('Errores')
plt.ylabel('Frecuencia')
plt.title('Distribución de Errores')
plt.show()












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




#### Analisis de sentimineto 

sentiment_analyzer = pipeline("sentiment-analysis")

# Función para calificar el texto en base al sentimiento
def rate_text(text):
    result = sentiment_analyzer(text)[0]
    if result['label'] == 'POSITIVE':
        # Escalar la puntuación en función de la confianza (score)
        return min(5, max(1, round(result['score'] * 5)))
    else:
        # Si es negativo, usar la puntuación inversa
        return min(5, max(1, round((1 - result['score']) * 5)))

# Reemplazar los valores NaN con una cadena vacía
df['summary'] = df['summary'].fillna("")

# Aplicar la función de calificación
df['calificacion'] = df['summary'].apply(rate_text)

print(df[['summary', 'calificacion']])

