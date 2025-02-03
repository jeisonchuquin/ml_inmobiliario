## INMOBILIARIA: MODELOS

En la carpeta **models** se encuentran 3 modelos que se testearon, el modelo que se utilizó está basado en `nlp` que se entrenó utilizando el servicio de **AWS Canvas**, sin embargo dentro de la carpeta **nlp_model** se encuentra el script utilizado para la limpieza y tratamiento de datos.

Los 3 modelos que se testearon son:

1. `nlp_model`
2. `images_model`
3. `finetuning_model`

Como objetivo de los modelos es que a partir de las características de entrada, `texto` en el caso de `nlp_model` y `finetuning_model` e `imágenes` en `images_model` se pueda predecir el precio de una casa.

#### MODELO NLP

Este modelo trabaja con todas las variables de texto, realizando el proceso clásico de transformación de palabras a vectores `word to vec`.

#### MODELO IMAGES

Las imágenes utilizadas se encuentran en el siguiente repositorio: [IMG](https://drive.google.com/drive/folders/1WPTqgfLeVKXxnHqYQ5RP9IL9qYPR_EKU?usp=drive_link)

Parte del preprocesamiento de normalización de imágenes se redimensionan las imágenes a `224 x 224` y se trabajan con las imágenes en `RGB`

Como modelo base se utilizó `VGG16` y luego se añade una capa de una sola neurona para poder predecir una variable continua.


#### MODELO FINETUNING

La construcción de este modelo está basado en el procedimiento de **Anthony Galtier** y que se puede encontrar en el siguiente [LINK](https://medium.com/ilb-labs-publications/fine-tuning-bert-for-a-regression-task-is-a-description-enough-to-predict-a-propertys-list-price-cf97cd7cb98a)

Se toma como modelo base `BETO` que es la versión en español de `BERT`, se puede descargar desde [HUGGING FACE](https://huggingface.co/dccuchile/bert-base-spanish-wwm-uncased) ya que las variables de texto son descripciones de casas que están en español.
