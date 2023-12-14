# Importar un modelo Keras a TensorFlow.js

Los modelos Keras (por lo común, creados con la API de Python) se pueden cargar en [uno de varios formatos](https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model).  El formato de "modelo entero" se puede convertir al de TensorFlow.js Layers, que da la posibilidad de cargarlo directamente en TensorFlow.js para realizar inferencias y continuar con el entrenamiento.

El formato TensorFlow.js Layers de destino es un directorio que contiene un archivo `model.json` y un conjunto de archivos de peso particionados horizontalmente en formato binario. El archivo `model.json` contiene tanto la topología del modelo (también conocida como "arquitectura" o "grafo": una descripción de las capas y de cómo están conectadas) y un manifiesto de los archivos de los pesos.

## Requisitos

Para el proceso de conversión se requiere de un entorno Python. Probablemente le resulte conveniente mantener uno aislado usando [pipenv](https://github.com/pypa/pipenv) o [virtualenv](https://virtualenv.pypa.io). Para instalar el conversor, use `pip install tensorflowjs`.

Importar un modelo Keras a TensorFlow.js es un proceso de dos pasos. Primero, hay que convertir un modelo Keras al formato TF.js Layers y luego hay que cargarlo en TensorFlow.js.

## Paso 1: convertir un modelo Keras al formato TF.js Layers

Los modelos Keras, por lo general, se guardan con `model.save(filepath)`, lo que produce un solo archivo HDF5 (.h5) que contiene tanto la topología como los pesos del modelo. Para convertir un archivo como este al formato TF.js Layers, ejecute el siguiente comando, donde *`path/to/my_model.h5`* es el archivo de origen Keras .h5 file y *`path/to/tfjs_target_dir`* es el directorio de salida de destino para los archivos TF.js:

```sh
# bash

tensorflowjs_converter --input_format keras \
                       path/to/my_model.h5 \
                       path/to/tfjs_target_dir
```

## Alternativa: uso de la API de Python para exportar directamente al formato TF.js Layers

Si tiene un modelo Keras en Python, se puede exportar directamente al formato TensorFlow.js Layers de la siguiente manera:

```py
# Python

import tensorflowjs as tfjs

def train(...):
    model = keras.models.Sequential()   # for example
    ...
    model.compile(...)
    model.fit(...)
    tfjs.converters.save_keras_model(model, tfjs_target_dir)
```

## Paso 2: cargar el modelo en TensorFlow.js

Use un servidor web para disponer de los archivos del modelo convertido, generados en el paso 1. Tenga en cuenta que puede necesitar configurar su servidor para [permitir el intercambio de recursos de origen cruzado (CORS)](https://enable-cors.org/) y facilitar la extracción de los archivos en JavaScript.

Después, cargue el modelo en TensorFlow.js proporcionando la URL al archivo del model.json:

```js
// JavaScript

import * as tf from '@tensorflow/tfjs';

const model = await tf.loadLayersModel('https://foo.bar/tfjs_artifacts/model.json');
```

Ahora, el modelo está listo para la inferencia, la evaluación o el reentrenamiento. Por ejemplo, el modelo cargado se puede usar inmediatamente para hacer una predicción:

```js
// JavaScript

const example = tf.fromPixels(webcamElement);  // for example
const prediction = model.predict(example);
```

Muchos de los [ejemplos de TensorFlow.js](https://github.com/tensorflow/tfjs-examples) toman este enfoque; usan modelos previamente entrenados que han sido convertidos y almacenados en Google Cloud.

Tenga en cuenta que al usar el nombre de archivo `model.json` nos referimos al modelo entero.  `loadModel(...)` extrae `model.json`, y después hace otras solicitudes HTTP(S) para obtener los archivos de peso particionados horizontalmente, referenciados en el manifiesto de pesos de `model.json`. Este método permite que el navegador guarde en caché todos estos archivos (y que tal vez otros servidores también lo hagan en internet), porque el `model.json` y las partes de los pesos particionados horizontalmente son más pequeñas que el límite de tamaño de un archivo de memoria caché típico. Por lo tanto, es probable que un modelo se cargue más rápido en las siguientes ocasiones.

## Características compatibles

En este momento, TensorFlow.js Layers solamente es compatible con los modelos Keras que usan construcciones Keras estándares. Los modelos que usan operaciones o capas que no son compatibles (p. ej., capas personalizadas, capas Lambda, pérdidas personalizadas o, incluso, métricas personalizadas) no se pueden importar automáticamente, porque dependen del código Python que no se puede traducir de manera confiable a JavaScript.
