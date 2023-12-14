# Segmentación

La segmentación de imágenes es el proceso de dividir una imagen digital en múltiples segmentos (conjuntos de pixeles, también conocidos como objetos de imagen). La meta de la segmentación es simplificar y/o cambiar la representación de una imagen en algo que sea más significativo y fácil de analizar.

La siguiente imagen muestra el resultado del modelo de segmentación de imágenes en Android. El modelo creará una máscara sobre los objetos objetivo con una alta precisión.

<img src="images/segmentation.gif" class="attempt-right">

Nota: Para integrar un modelo existente, pruebe la [librería de tareas de TensorFlow Lite](https://www.tensorflow.org/lite/inference_with_metadata/task_library/image_segmenter).

## Empecemos

Si es nuevo en TensorFlow Lite y trabaja con Android o iOS, se recomienda explorar las siguientes aplicaciones de ejemplo que pueden ayudarle a empezar.

Puede aprovechar la API lista para usar de la [Librería de tareas de TensorFlow Lite](../../inference_with_metadata/task_library/image_segmenter) para integrar modelos de segmentación de imágenes en unas pocas líneas de código. También puede integrar el modelo usando la [API Java del intérprete de TensorFlow Lite](../../guide/inference#load_and_run_a_model_in_java).

El siguiente ejemplo de Android muestra la implementación de ambos métodos como [lib_task_api](https://github.com/tensorflow/examples/tree/master/lite/examples/image_segmentation/android/lib_task_api) y [lib_interpreter](https://github.com/tensorflow/examples/tree/master/lite/examples/image_segmentation/android/lib_interpreter), respectivamente.

<a class="button button-primary" href="https://github.com/tensorflow/examples/tree/master/lite/examples/image_segmentation/android">Ver ejemplo en Android</a>

<a class="button button-primary" href="https://github.com/tensorflow/examples/tree/master/lite/examples/image_segmentation/ios">Ver ejemplo en iOS</a>

Si utiliza una plataforma distinta de Android o iOS o ya está familiarizado con las <a href="https://www.tensorflow.org/api_docs/python/tf/lite">API de TensorFlow Lite</a>, puede descargar nuestro modelo de recomendación para principiantes.

<a class="button button-primary" href="https://tfhub.dev/tensorflow/lite-model/deeplabv3/1/metadata/2?lite-format=tflite">Descargue el modelo inicial</a>

## Descripción del modelo

*DeepLab* es un modelo de aprendizaje profundo de última generación para la segmentación semántica de imágenes, en el que la meta es asignar etiquetas semánticas (por ejemplo, persona, perro, gato) a cada pixel de la imagen de entrada.

### Cómo funciona

La segmentación semántica de imágenes predice si cada pixel de una imagen está asociado a una clase determinada. Esto contrasta con la <a href="../object_detection/overview.md">detección de objetos</a>, que detecta objetos en regiones rectangulares, y la <a href="../image_classification/overview.md">clasificación de imágenes</a>, que clasifica la imagen en su conjunto.

La implementación actual incluye las siguientes características:

<ol>
  <li>DeepLabv1: Usamos la convolución atrófica para controlar explícitamente la resolución con la que se calculan las respuestas a las características dentro de las redes neuronales convolucionales profundas.</li>
  <li>DeepLabv2: Usamos la agrupación de pirámides espaciales atróficas (ASPP) para segmentar de forma robusta objetos a múltiples escalas con filtros a múltiples velocidades de muestreo y campos de visión efectivos.</li>
  <li>DeepLabv3: Aumentamos el módulo ASPP con características a nivel de imagen [5, 6] para captar información de mayor alcance. También incluimos parámetros de normalización por lotes [7] para facilitar el entrenamiento. En particular, aplicamos la convolución atrófica para extraer las características de salida en diferentes pasos de salida durante el entrenamiento y la evaluación, lo que permite entrenar de forma eficiente el BN en el paso de salida = 16 y alcanzar un alto rendimiento en el paso de salida = 8 durante la evaluación.</li>
  <li>DeepLabv3+: Ampliamos DeepLabv3 para incluir un módulo decodificador sencillo pero eficaz para refinar los resultados de la segmentación, especialmente a lo largo de los límites de los objetos. Además, en esta estructura codificador-decodificador se puede controlar arbitrariamente la resolución de las características extraídas del codificador mediante convolución atrófica para compensar precisión y runtime.</li>
</ol>

## Puntos de referencia del rendimiento

Los números de referencia del rendimiento se generan con la herramienta [descrita aquí](https://www.tensorflow.org/lite/performance/benchmarks).

<table>
  <thead>
    <tr>
      <th>Nombre del modelo</th>
      <th>Tamaño del modelo</th>
      <th>Dispositivo</th>
      <th>GPU</th>
      <th>CPU</th>
    </tr>
  </thead>
  <tr>
    <td rowspan="3">       <a href="https://tfhub.dev/tensorflow/lite-model/deeplabv3/1/metadata/2?lite-format=tflite">Deeplab v3</a>
</td>
    <td rowspan="3">       2.7 Mb</td>
    <td>Pixel 3 (Android 10)</td>
    <td>16 ms</td>
    <td>37 ms*</td>
  </tr>
   <tr>
     <td>Pixel 4 (Android 10)</td>
    <td>20 ms</td>
    <td>23 ms*</td>
  </tr>
   <tr>
     <td>iPhone XS (iOS 12.4.1)</td>
     <td>16 ms</td>
    <td>25 ms**</td>
  </tr>
</table>

* 4 hilos usados.

** 2 hilos usados en el iPhone para obtener el mejor resultado de rendimiento.

## Lecturas y recursos complementarios

<ul>
  <li><a href="https://ai.googleblog.com/2018/03/semantic-image-segmentation-with.html">Segmentación semántica de imágenes con DeepLab en TensorFlow</a></li>
  <li><a href="https://medium.com/tensorflow/tensorflow-lite-now-faster-with-mobile-gpus-developer-preview-e15797e6dee7">TensorFlow Lite ahora más rápido con GPUs móviles (Vista previa para desarrolladores)</a></li>
  <li><a href="https://github.com/tensorflow/models/tree/master/research/deeplab">DeepLab: Etiquetado profundo para la segmentación semántica de imágenes</a></li>
</ul>
