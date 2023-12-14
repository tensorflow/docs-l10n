# Clasificación de texto

Usar un modelo TensorFlow Lite para categorizar un párrafo en grupos predefinidos.

Nota: (1) Para integrar un modelo existente, pruebe la [Librería de tareas de TensorFlow Lite](https://www.tensorflow.org/lite/inference_with_metadata/task_library/nl_classifier). (2) Para personalizar un modelo, pruebe [Model Maker de TensorFlow Lite](https://www.tensorflow.org/lite/models/modify/model_maker/text_classification).

## Empecemos

<img src="images/screenshot.gif" class="attempt-right" style="max-width: 300px">

Si no tiene experiencia con TensorFlow Lite y trabaja con Android, le recomendamos que explore la guía de [Librería de tareas de TensorFLow Lite](../../inference_with_metadata/task_library/nl_classifier) para integrar modelos de clasificación de texto en unas pocas líneas de código. También puede integrar el modelo usando la [API Java del intérprete de TensorFlow Lite](../../guide/inference#load_and_run_a_model_in_java).

El siguiente ejemplo de Android muestra la implementación de ambos métodos como [lib_task_api](https://github.com/tensorflow/examples/tree/master/lite/examples/text_classification/android/lib_task_api) y [lib_interpreter](https://github.com/tensorflow/examples/tree/master/lite/examples/text_classification/android/lib_interpreter), respectivamente.

<a class="button button-primary" href="https://github.com/tensorflow/examples/tree/master/lite/examples/text_classification/android">Ejemplo en Android</a>

Si utiliza una plataforma distinta de Android o ya está familiarizado con las API de TensorFlow Lite, puede descargar nuestro modelo inicial de clasificación de texto.

<a class="button button-primary" href="https://storage.googleapis.com/download.tensorflow.org/models/tflite/text_classification/text_classification_v2.tflite">Descargue el modelo inicial</a>

## Cómo funciona

La clasificación de textos clasifica un párrafo en grupos predefinidos en función de su contenido.

Este modelo preentrenado predice si el sentimiento de un párrafo es positivo o negativo. Se entrenó con [Large Movie Review Dataset v1.0](http://ai.stanford.edu/~amaas/data/sentiment/) de Mass et al, que consiste en reseñas de películas de IMDB etiquetadas como positivas o negativas.

Estos son los pasos para clasificar un párrafo con el modelo:

1. Tokeniza el párrafo y lo convierte en una lista de ids de palabras usando un vocabulario predefinido.
2. Introduzca la lista en el modelo TensorFlow Lite.
3. Obtenga la probabilidad de que el párrafo sea positivo o negativo a partir de los resultados del modelo.

### Nota

- Sólo es compatible con el inglés.
- Este modelo se entrenó en un conjunto de datos de críticas de películas, por lo que puede experimentar una precisión reducida al clasificar texto de otros dominios.

## Puntos de referencia del rendimiento

Los números de referencia del rendimiento se generan con la herramienta [descrita aquí](https://www.tensorflow.org/lite/performance/benchmarks).

<table>
  <thead>
    <tr>
      <th>Nombre del modelo</th>
      <th>Tamaño del modelo</th>
      <th>Dispositivo</th>
      <th>CPU</th>
    </tr>
  </thead>
  <tr>
    <td rowspan="3">       <a href="https://storage.googleapis.com/download.tensorflow.org/models/tflite/text_classification/text_classification_v2.tflite">Clasificación de texto</a>
</td>
    <td rowspan="3">       0.6 Mb</td>
    <td>Pixel 3 (Android 10)</td>
    <td>0.05 ms*</td>
  </tr>
   <tr>
     <td>Pixel 4 (Android 10)</td>
    <td>0.05 ms*</td>
  </tr>
   <tr>
     <td>iPhone XS (iOS 12.4.1)</td>
    <td>0.025 ms**</td>
  </tr>
</table>

* 4 hilos usados.

** 2 hilos usados en el iPhone para obtener el mejor resultado de rendimiento.

## Salida de ejemplo

Texto | Negativo (0) | Positivo (1)
--- | --- | ---
Esta es la mejor película que he visto en los últimos | 25.3 % | 74.7 %
: años. ¡Muy recomendable!              :              :              : |  |
Qué pérdida de tiempo. | 72.5 % | 27.5 %

## Usar su conjunto de datos de entrenamiento

Siga este [tutorial](https://www.tensorflow.org/lite/models/modify/model_maker/text_classification) para aplicar la misma técnica utilizada aquí para usar en el entrenamiento de un modelo de clasificación de texto usando sus propios conjuntos de datos. Con el conjunto de datos adecuado, podrá crear un modelo para casos de uso como la categorización de documentos o la detección de comentarios tóxicos.

## Más información sobre la clasificación de textos

- [Incorporaciones de palabras y tutorial para entrenar este modelo](https://www.tensorflow.org/tutorials/text/word_embeddings)
