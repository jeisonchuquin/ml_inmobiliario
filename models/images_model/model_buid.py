import os
import numpy as np
import pandas as pd
from tensorflow.keras.applications import VGG16
from sklearn.model_selection import train_test_split
from PIL import Image
from tensorflow.keras import layers, models


def preprocess_image(img, target_size=(224, 224)):    
    '''
    Función para preprocesar las imágenes (redimensionar y normalizar)
    '''
    
    img = img.resize(target_size)  # Redimensionar la imagen a 224x224
    img = img.convert('RGB')  # Convertir todas las imágenes a RGB (3 canales)
    img = np.array(img) / 255.0  # Normalizar la imagen

    return img

def load_images_from_folder(folder):
    '''
    Función para cargar imágenes de una carpeta
    '''
    
    images = []
    
    for filename in os.listdir(folder):
    
        try:
    
            img = Image.open(os.path.join(folder, filename))  # Abrir la imagen
    
            if img is not None:
    
                img = preprocess_image(img)  # Preprocesar la imagen
                images.append(np.array(img))
    
        except (IOError, SyntaxError) as e:
    
            print(f"Error loading image {filename}: {e}")
    
    return images


#! Lectura de datos
monto = pd.read_csv(r'data\raw\data_inmobiliario.csv', encoding='latin1')


# Convertir las columnas CODIGO y PRECIO a listas
codigos = monto['CODIGO'].tolist()
precios = monto['PRECIO'].tolist()


# Escalar los precios dividiéndolos por 1 millón
precios = [precio / 1_000_000 for precio in precios]

# Crear el diccionario de precios con el código como clave
house_prices = dict(zip(codigos, precios))

# Leer imágenes desde las carpetas (cada carpeta tiene imágenes de casas)
# imágenes están en el link de gdrive
carpetas = '/content/drive/MyDrive/IMG'
data = {}
contador = 0
limite_carpetas = 180  # Límite de carpetas a procesar

imagenes = []
etiquetas = []

# Recorrer las carpetas de imágenes
for carpeta in os.listdir(carpetas):
    
    if contador >= limite_carpetas:
        break
    
    folder_path = os.path.join(carpetas, carpeta)
    if os.path.isdir(folder_path):
        
        images = load_images_from_folder(folder_path)
        
        if carpeta in house_prices:
            precio = house_prices[carpeta]
            
            if images:
                imagenes.extend(images)                
                etiquetas.extend([precio] * len(images))
        else:
            
            print(
                f"Advertencia: No se encontró precio para la carpeta {carpeta}. Las imágenes no serán añadidas."
            )

        contador += 1


# Convertir las listas a arrays de NumPy
imagenes = np.array(imagenes)
etiquetas = np.array(etiquetas)

# Comprobación del número de imágenes y etiquetas
print(f"Total de imágenes: {len(imagenes)}")
print(f"Total de etiquetas: {len(etiquetas)}")

# Verificar si hay una diferencia en el número de imágenes y etiquetas
if len(imagenes) != len(etiquetas):
    raise ValueError(
        f"Inconsistencia en el número de imágenes ({len(imagenes)}) y etiquetas ({len(etiquetas)})."
    )

#! Primera metodología

# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    imagenes, etiquetas, 
    test_size=0.2, 
    random_state=42
)


# Utilizar VGG16 como red preentrenada
base_model = VGG16(
    weights='imagenet', 
    include_top=False,
    input_shape=(224, 224, 3)
)
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
])


# Compilar el modelo con el optimizador Adam y la pérdida de MSE
model.compile(optimizer='adam', loss='mean_squared_error')

# Entrenar el modelo con las imágenes y los precios
history = model.fit(
    X_train, 
    y_train, 
    epochs=10,
    batch_size=32, 
    validation_data=(X_test, y_test)
)

# Evaluar el modelo en los datos de prueba
test_loss = model.evaluate(X_test, y_test)
print(f"Loss en los datos de prueba: {test_loss}")


#! Segunda metodología

# Comprobación del número de imágenes y etiquetas
print(f"Total de imágenes: {len(imagenes)}")
print(f"Total de etiquetas: {len(etiquetas)}")

# Verificar si hay una diferencia en el número de imágenes y etiquetas
if len(imagenes) != len(etiquetas):
    raise ValueError(
        f"Inconsistencia en el número de imágenes ({len(imagenes)}) y etiquetas ({len(etiquetas)})."
    )

# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(imagenes, etiquetas, test_size=0.2, random_state=42)

# Definir el modelo CNN
model = models.Sequential(
    [
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
history = model.fit(
    X_train, 
    y_train, 
    epochs=10, 
    batch_size=32, 
    validation_data=(X_test, y_test)
)

# Evaluar el modelo en los datos de prueba
test_loss = model.evaluate(X_test, y_test)
print(f"Loss en los datos de prueba: {test_loss}")