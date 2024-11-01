
import os
from PIL import Image
import numpy as np

# Función para cargar imágenes de una carpeta
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        try:
            img = Image.open(os.path.join(folder, filename))  # Abrir la imagen
            if img is not None:
                img = preprocess_image(img)  # Preprocesar la imagen
                images.append(np.array(img))
        except (IOError, SyntaxError) as e:  # Manejo de excepciones
            print(f"Error loading image {filename}: {e}")  # Imprimir error
    return images



# Leer imágenes desde las carpetas (cada carpeta tiene imágenes de casas)
carpetas = '/content/drive/MyDrive/IMG'
data = {}
contador = 0
limite_carpetas = 180  # Límite de carpetas a procesar

"""- Con CPU se murió el kernel
- Con TPU se ejecuta con más de 57 min

- Error loading image img_0.png: cannot identify image file '/content/drive/MyDrive/IMG/EC-RJ-133445/img_0.png'
- Error loading image img_1.png: cannot identify image file '/content/drive/MyDrive/IMG/EC-RJ-133445/img_1.png'
- Error loading image img_2.png: cannot identify image file '/content/drive/MyDrive/IMG/EC-RJ-133445/img_2.png'
"""

# Función para preprocesar las imágenes (redimensionar y normalizar)
def preprocess_image(img, target_size=(224, 224)):
    img = img.resize(target_size)  # Redimensionar la imagen a 224x224
    img = img.convert('RGB')  # Convertir todas las imágenes a RGB (3 canales)
    img = np.array(img) / 255.0  # Normalizar la imagen
    return img

"""#MONTO"""

import openpyxl

import pandas as pd

# Leer el archivo Excel con los precios
monto = pd.read_excel('/content/drive/MyDrive/monto.xlsx')
monto.shape

# Convertir las columnas CODIGO y PRECIO a listas
codigos = monto['CODIGO'].tolist()
precios = monto['PRECIO'].tolist()

# Escalar los precios dividiéndolos por 1 millón
precios = [precio / 1000000 for precio in precios]

# Crear el diccionario de precios con el código como clave
house_prices = dict(zip(codigos, precios))

"""#MONTO+IMAGEN"""

imagenes = []
etiquetas = []

# Recorrer las carpetas de imágenes
for carpeta in os.listdir(carpetas):
    if contador >= limite_carpetas:
        break  # Salir del bucle si ya se ha alcanzado el límite
    folder_path = os.path.join(carpetas, carpeta)
    if os.path.isdir(folder_path):  # Verificar si es una carpeta
        images = load_images_from_folder(folder_path)

        # Verificar si el código de la carpeta tiene un precio correspondiente
        if carpeta in house_prices:
            precio = house_prices[carpeta]
            if images:  # Si hay imágenes cargadas, las añadimos a los datos
                imagenes.extend(images)  # Añadir todas las imágenes a la lista
                etiquetas.extend([precio] * len(images))  # Mismo precio para todas las imágenes de la casa
        else:
            print(f"Advertencia: No se encontró precio para la carpeta {carpeta}. Las imágenes no serán añadidas.")

        contador += 1

# Convertir las listas a arrays de NumPy
imagenes = np.array(imagenes)
etiquetas = np.array(etiquetas)

# Comprobación del número de imágenes y etiquetas
print(f"Total de imágenes: {len(imagenes)}")
print(f"Total de etiquetas: {len(etiquetas)}")

# Verificar si hay una diferencia en el número de imágenes y etiquetas
#if len(imagenes) != len(etiquetas):
#    raise ValueError(f"Inconsistencia en el número de imágenes ({len(imagenes)}) y etiquetas ({len(etiquetas)}).")

"""#PARTICION DE LA DATA"""

# Dividir los datos en entrenamiento y prueba
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(imagenes, etiquetas, test_size=0.2, random_state=2024)

"""#RED NEURONAL"""

from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models

# Utilizar VGG16 como red preentrenada
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Congelamos las capas preentrenadas

model = models.Sequential([
    base_model,
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)  # Salida para el precio de la casa
])

"""#OPTIMIZADOR ADAM"""

# Compilar el modelo con el optimizador Adam y la pérdida de MSE
model.compile(optimizer='adam', loss='mean_squared_error')

"""#ENTRENAMIENTO"""

# Entrenar el modelo con las imágenes y los precios
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

"""#PRUEBA"""

# Evaluar el modelo en los datos de prueba
test_loss = model.evaluate(X_test, y_test)
print(f"Loss en los datos de prueba: {test_loss}")

"""#CODIGO COMPLETO"""

import os
from PIL import Image
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models

# Función para preprocesar las imágenes (redimensionar y normalizar)
def preprocess_image(img, target_size=(224, 224)):
    img = img.resize(target_size)  # Redimensionar la imagen a 224x224
    img = img.convert('RGB')  # Convertir todas las imágenes a RGB (3 canales)
    img = np.array(img) / 255.0  # Normalizar la imagen
    return img

# Función para cargar imágenes de una carpeta
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        try:
            img = Image.open(os.path.join(folder, filename))  # Abrir la imagen
            if img is not None:
                img = preprocess_image(img)  # Preprocesar la imagen
                images.append(np.array(img))
        except (IOError, SyntaxError) as e:  # Manejo de excepciones
            print(f"Error loading image {filename}: {e}")  # Imprimir error
    return images

# Leer el archivo Excel con los precios
monto = pd.read_excel('/content/drive/MyDrive/monto.xlsx')

# Convertir las columnas CODIGO y PRECIO a listas
codigos = monto['CODIGO'].tolist()
precios = monto['PRECIO'].tolist()
# Escalar los precios dividiéndolos por 1 millón
precios = [precio / 1_000_000 for precio in precios]


# Crear el diccionario de precios con el código como clave
house_prices = dict(zip(codigos, precios))

# Leer imágenes desde las carpetas (cada carpeta tiene imágenes de casas)
carpetas = '/content/drive/MyDrive/IMG'
data = {}
contador = 0
limite_carpetas = 180  # Límite de carpetas a procesar

imagenes = []
etiquetas = []

# Recorrer las carpetas de imágenes
for carpeta in os.listdir(carpetas):
    if contador >= limite_carpetas:
        break  # Salir del bucle si ya se ha alcanzado el límite
    folder_path = os.path.join(carpetas, carpeta)
    if os.path.isdir(folder_path):  # Verificar si es una carpeta
        images = load_images_from_folder(folder_path)

        # Verificar si el código de la carpeta tiene un precio correspondiente
        if carpeta in house_prices:
            precio = house_prices[carpeta]
            if images:  # Si hay imágenes cargadas, las añadimos a los datos
                imagenes.extend(images)  # Añadir todas las imágenes a la lista
                etiquetas.extend([precio] * len(images))  # Mismo precio para todas las imágenes de la casa
        else:
            print(f"Advertencia: No se encontró precio para la carpeta {carpeta}. Las imágenes no serán añadidas.")

        contador += 1

# Convertir las listas a arrays de NumPy
imagenes = np.array(imagenes)
etiquetas = np.array(etiquetas)

# Comprobación del número de imágenes y etiquetas
print(f"Total de imágenes: {len(imagenes)}")
print(f"Total de etiquetas: {len(etiquetas)}")

# Verificar si hay una diferencia en el número de imágenes y etiquetas
if len(imagenes) != len(etiquetas):
    raise ValueError(f"Inconsistencia en el número de imágenes ({len(imagenes)}) y etiquetas ({len(etiquetas)}).")

# Dividir los datos en entrenamiento y prueba
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(imagenes, etiquetas, test_size=0.2, random_state=42)


from tensorflow.keras.applications import VGG16

# Utilizar VGG16 como red preentrenada
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Congelamos las capas preentrenadas

model = models.Sequential([
    base_model,
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)  # Salida para el precio de la casa
])


# Compilar el modelo con el optimizador Adam y la pérdida de MSE
model.compile(optimizer='adam', loss='mean_squared_error')

# Entrenar el modelo con las imágenes y los precios
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Evaluar el modelo en los datos de prueba
test_loss = model.evaluate(X_test, y_test)
print(f"Loss en los datos de prueba: {test_loss}")

"""#BORRADOR"""

import os
from PIL import Image
import numpy as np

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        try:
            img = Image.open(os.path.join(folder, filename))  # Try to open the image
            if img is not None:
                images.append(np.array(img))
        except (IOError, SyntaxError) as e:  # Handle potential exceptions like file not found or invalid image data
            print(f"Error loading image {filename}: {e}")  # Print an error message with the filename
    return images

# Ejemplo de lectura por carpeta
carpetas = '/content/drive/MyDrive/IMG'
data = {}
contador = 0
limite_carpetas = 180  # Límite de carpetas a procesar

for carpeta in os.listdir(carpetas):
    if contador >= limite_carpetas:
        break  # Salir del bucle si ya se ha alcanzado el límite
    folder_path = os.path.join(carpetas, carpeta)
    if os.path.isdir(folder_path):  # Verifica si es una carpeta
        images = load_images_from_folder(folder_path)
        data[carpeta] = images
        contador += 1

import os
from PIL import Image
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models

# Función para preprocesar las imágenes (redimensionar y normalizar)
def preprocess_image(img, target_size=(224, 224)):
    img = img.resize(target_size)  # Redimensionar la imagen a 224x224
    img = img.convert('RGB')  # Convertir todas las imágenes a RGB (3 canales)
    img = np.array(img) / 255.0  # Normalizar la imagen
    return img

# Función para cargar imágenes de una carpeta
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        try:
            img = Image.open(os.path.join(folder, filename))  # Abrir la imagen
            if img is not None:
                img = preprocess_image(img)  # Preprocesar la imagen
                images.append(np.array(img))
        except (IOError, SyntaxError) as e:  # Manejo de excepciones
            print(f"Error loading image {filename}: {e}")  # Imprimir error
    return images

# Leer el archivo Excel con los precios
monto = pd.read_excel('/content/drive/MyDrive/monto.xlsx')

# Convertir las columnas CODIGO y PRECIO a listas
codigos = monto['CODIGO'].tolist()
precios = monto['PRECIO'].tolist()
# Escalar los precios dividiéndolos por 1 millón
precios = [precio / 1_000_000 for precio in precios]


# Crear el diccionario de precios con el código como clave
house_prices = dict(zip(codigos, precios))

# Leer imágenes desde las carpetas (cada carpeta tiene imágenes de casas)
carpetas = '/content/drive/MyDrive/IMG'
data = {}
contador = 0
limite_carpetas = 180  # Límite de carpetas a procesar

imagenes = []
etiquetas = []

# Recorrer las carpetas de imágenes
for carpeta in os.listdir(carpetas):
    if contador >= limite_carpetas:
        break  # Salir del bucle si ya se ha alcanzado el límite
    folder_path = os.path.join(carpetas, carpeta)
    if os.path.isdir(folder_path):  # Verificar si es una carpeta
        images = load_images_from_folder(folder_path)

        # Verificar si el código de la carpeta tiene un precio correspondiente
        if carpeta in house_prices:
            precio = house_prices[carpeta]
            if images:  # Si hay imágenes cargadas, las añadimos a los datos
                imagenes.extend(images)  # Añadir todas las imágenes a la lista
                etiquetas.extend([precio] * len(images))  # Mismo precio para todas las imágenes de la casa
        else:
            print(f"Advertencia: No se encontró precio para la carpeta {carpeta}. Las imágenes no serán añadidas.")

        contador += 1

# Convertir las listas a arrays de NumPy
imagenes = np.array(imagenes)
etiquetas = np.array(etiquetas)

# Comprobación del número de imágenes y etiquetas
print(f"Total de imágenes: {len(imagenes)}")
print(f"Total de etiquetas: {len(etiquetas)}")

# Verificar si hay una diferencia en el número de imágenes y etiquetas
if len(imagenes) != len(etiquetas):
    raise ValueError(f"Inconsistencia en el número de imágenes ({len(imagenes)}) y etiquetas ({len(etiquetas)}).")

# Dividir los datos en entrenamiento y prueba
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(imagenes, etiquetas, test_size=0.2, random_state=42)

# Definir el modelo CNN
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)  # Salida para el precio de la casa
])

# Compilar el modelo con el optimizador Adam y la pérdida de MSE
model.compile(optimizer='adam', loss='mean_squared_error')

# Entrenar el modelo con las imágenes y los precios
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Evaluar el modelo en los datos de prueba
test_loss = model.evaluate(X_test, y_test)
print(f"Loss en los datos de prueba: {test_loss}")

# Ejemplo de lectura por carpeta
carpetas = '/content/drive/MyDrive/IMG'
data = {}
for carpeta in os.listdir(carpetas):
    folder_path = os.path.join(carpetas, carpeta)
    images = load_images_from_folder(folder_path)
    data[carpeta] = images