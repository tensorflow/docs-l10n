# Clasificación de imágenes

<img src="../images/image.png" class="attempt-right">

La tarea de identificar lo que representa una imagen se denomina *clasificación de imágenes*. Un modelo de clasificación de imágenes se entrena para reconocer varias clases de imágenes. Por ejemplo, puede entrenar un modelo para reconocer fotos que representen tres tipos diferentes de animales: conejos, hámsters y perros. TensorFlow Lite proporciona modelos preentrenados optimizados que puede implementar en sus aplicaciones móviles. Obtenga más información sobre la clasificación de imágenes usando TensorFlow [aquí](https://www.tensorflow.org/tutorials/images/classification).

La siguiente imagen muestra la salida del modelo de clasificación de imágenes en Android.

<img src="images/android_banana.png" width="30%" alt="Captura de pantalla de ejemplo de Android">

Nota: (1) Para integrar un modelo existente, pruebe la [Librería de tareas de TensorFlow Lite](https://www.tensorflow.org/lite/inference_with_metadata/task_library/image_classifier). (2) Para personalizar un modelo, pruebe [Model Maker de TensorFlow Lite](https://www.tensorflow.org/lite/models/modify/model_maker/image_classification).

## Empecemos

Si es nuevo en TensorFlow Lite y trabaja con Android o iOS, se recomienda explorar las siguientes aplicaciones de ejemplo que pueden ayudarle a empezar.

Puede aprovechar la API lista para usar de la [Librería de tareas TensorFlow Lite](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/g3doc/inference_with_metadata/task_library/image_classifier.md) para integrar modelos de clasificación de imágenes en tan solo unas líneas de código. También puede construir su propia canalización de inferencia personalizada usando la [Librería de soporte de TensorFlow Lite](../../inference_with_metadata/lite_support).

El siguiente ejemplo de Android muestra la implementación de ambos métodos como [lib_task_api](https://github.com/tensorflow/examples/tree/master/lite/examples/image_classification/android/lib_task_api) y [lib_support](https://github.com/tensorflow/examples/tree/master/lite/examples/image_classification/android/lib_support), respectivamente.

<a class="button button-primary" href="https://github.com/tensorflow/examples/tree/master/lite/examples/image_classification/android">Ver ejemplo en Android</a>

<a class="button button-primary" href="https://github.com/tensorflow/examples/tree/master/lite/examples/image_classification/ios">Ver ejemplo en iOS</a>

Si utiliza una plataforma distinta de Android/iOS, o si ya está familiarizado con las API de [TensorFlow Lite](https://www.tensorflow.org/api_docs/python/tf/lite), descargue el modelo de inicio y los archivos de soporte (si procede).

<a class="button button-primary" href="https://storage.googleapis.com/download.tensorflow.org/models/tflite/mobilenet_v1_1.0_224_quant_and_labels.zip">Descargue el modelo inicial</a>

## Descripción del modelo

### Cómo funciona

Durante el entrenamiento, un modelo de clasificación de imágenes recibe imágenes y sus *etiquetas* asociadas. Cada etiqueta es el nombre de un concepto distinto, o clase, que el modelo aprenderá a reconocer.

Dados suficientes datos de entrenamiento (a menudo cientos o miles de imágenes por etiqueta), un modelo de clasificación de imágenes puede aprender a predecir si las nuevas imágenes pertenecen a alguna de las clases en las que ha sido entrenado. Este proceso de predicción se denomina *inferencia*. Tenga en cuenta que también puede usar el [aprendizaje por transferencia](https://www.tensorflow.org/tutorials/images/transfer_learning) para identificar nuevas clases de imágenes usando un modelo preexistente. El aprendizaje por transferencia no requiere un conjunto de datos de entrenamiento muy grande.

Cuando posteriormente proporcione una nueva imagen como entrada al modelo, éste emitirá las probabilidades de que la imagen represente cada uno de los tipos de animales con los que fue entrenado. Un ejemplo de salida podría ser el siguiente:

<table style="width: 40%;">
  <thead>
    <tr>
      <th>Tipo de animal</th>
      <th>Probabilidad</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Conejo</td>
      <td>0.07</td>
    </tr>
    <tr>
      <td>Hamster</td>
      <td>0.02</td>
    </tr>
    <tr>
      <td style="background-color: #fcb66d;">Perro</td>
      <td style="background-color: #fcb66d;">0.91</td>
    </tr>
  </tbody>
</table>

Cada número de la salida corresponde a una etiqueta de los datos del entrenamiento. Asociando la salida con las tres etiquetas con las que se entrenó el modelo, se puede ver que el modelo ha predicho una alta probabilidad de que la imagen represente a un perro.

Puede que note que la suma de todas las probabilidades (para conejo, hámster y perro) es igual a 1. Este es un tipo de salida común para los modelos con múltiples clases (consulte <a href="https://developers.google.com/machine-learning/crash-course/multi-class-neural-networks/softmax">Softmax</a> para obtener más información).

Nota: La clasificación de imágenes sólo puede indicarle la probabilidad de que una imagen represente una o varias de las clases sobre las que se ha entrenado el modelo. No puede decirle la posición o identidad de los objetos dentro de la imagen. Si necesita identificar objetos y sus posiciones dentro de las imágenes, debe usar un modelo de <a href="../object_detection/overview">detección de objetos</a>.

<h4>Resultados ambiguos</h4>

Dado que las probabilidades de salida siempre sumarán 1, si una imagen no se reconoce con seguridad como perteneciente a ninguna de las clases en las que se ha entrenado el modelo, es posible que la probabilidad se distribuya entre las etiquetas sin que ningún valor sea significativamente mayor.

Por ejemplo, lo siguiente podría indicar un resultado ambiguo:


<table style="width: 40%;">   <thead>     <tr>       <th>Etiqueta</th>       <th>Probabilidad</th>     </tr>   </thead>   <tbody>     <tr>       <td>conejo</td>       <td>0.31</td>     </tr>     <tr>       <td>hamster</td>       <td>0.35</td>     </tr>     <tr>       <td>perro</td>       <td>0.34</td>     </tr>   </tbody> </table> Si su modelo devuelve con frecuencia resultados ambiguos, es posible que necesite un modelo diferente y más preciso.

<h3>Seleccionar una arquitectura modelo</h3>

TensorFlow Lite le ofrece una gran variedad de modelos de clasificación de imágenes, todos ellos entrenados en el conjunto de datos original. Hay disponibles arquitecturas de modelos como MobileNet, Inception y NASNet en <a href="https://tfhub.dev/s?deployment-format=lite">TensorFlow Hub</a>. Para elegir el mejor modelo para su caso de uso, debe tener en cuenta las arquitecturas individuales, así como algunas de las compensaciones entre varios modelos. Algunas de estas compensaciones entre modelos se basan en métricas como el rendimiento, la precisión y el tamaño del modelo. Por ejemplo, puede que necesite un modelo más rápido para construir un escáner de códigos de barras, mientras que puede que prefiera un modelo más lento y preciso para una app de imágenes médicas.

Tenga en cuenta que los <a href="https://www.tensorflow.org/lite/guide/hosted_models#image_classification">modelos de clasificación de imágenes</a> proporcionados aceptan distintos tamaños de entrada. Para algunos modelos, esto se indica en el nombre del archivo. Por ejemplo, el modelo Mobilenet_V1_1.0_224 acepta una entrada de 224x224 pixel. Todos los modelos requieren tres canales de color por pixel (rojo, verde y azul). Los modelos cuantizados requieren 1 byte por canal, y los modelos flotantes requieren 4 bytes por canal. Los ejemplos de código <a href="https://github.com/tensorflow/examples/tree/master/lite/examples/image_classification/android_java">Android</a> e <a href="https://github.com/tensorflow/examples/tree/master/lite/examples/image_classification/ios">iOS</a> demuestran cómo procesar imágenes de cámara de tamaño completo en el formato requerido para cada modelo.

<h3>Usos y limitaciones</h3>

Los modelos de clasificación de imágenes de TensorFlow Lite son útiles para la clasificación de una sola etiqueta; es decir, para predecir qué etiqueta única es más probable que represente la imagen. Están entrenados para reconocer 1000 clases de imágenes. Para obtener una lista completa de clases, consulte el archivo de etiquetas en el <a href="https://storage.googleapis.com/download.tensorflow.org/models/tflite/mobilenet_v1_1.0_224_quant_and_labels.zip">archivo comprimido zip del modelo</a>.

Si desea entrenar un modelo para reconocer nuevas clases, consulte <a href="#customize_model">Personalizar modelo</a>.

Para los siguientes casos de uso, deberá usar un tipo de modelo diferente:

<ul>
  <li>Predecir el tipo y la posición de uno o varios objetos dentro de una imagen (consulte <a href="../object_detection/overview">Detección de objetos</a>)</li>
  <li>Predecir la composición de una imagen, por ejemplo sujeto frente a fondo (consulte <a href="../segmentation/overview">Segmentación</a>)</li>
</ul>

Una vez que tenga el modelo inicial funcionando en su dispositivo objetivo, puede experimentar con diferentes modelos para encontrar el equilibrio óptimo entre rendimiento, precisión y tamaño del modelo.

<h3>Personalizar el modelo</h3>

Los modelos preentrenados proporcionados están entrenados para reconocer 1000 clases de imágenes. Para obtener una lista completa de las clases, consulte el archivo de etiquetas en el <a href="https://storage.googleapis.com/download.tensorflow.org/models/tflite/mobilenet_v1_1.0_224_quant_and_labels.zip">archivo zip del modelo</a>.

También puede usar el aprendizaje por transferencia para volver a entrenar un modelo para que reconozca clases que no están en el conjunto original. Por ejemplo, podría volver a entrenar el modelo para distinguir entre distintas especies de árboles, a pesar de que no hubiera árboles en los datos de entrenamiento originales. Para ello, necesitará un conjunto de imágenes de entrenamiento para cada una de las nuevas etiquetas que desee entrenar.

Aprenda a realizar el aprendizaje por transferencia con el <a href="https://www.tensorflow.org/lite/models/modify/model_maker/image_classification">TFLite Model Maker</a>, o en el <a href="https://codelabs.developers.google.com/codelabs/recognize-flowers-with-tensorflow-on-android/index.html#0">Codelab de reconocimiento de flores con TensorFlow</a>.

<h2>Puntos de referencia del rendimiento</h2>

El rendimiento del modelo se mide en términos de la cantidad de tiempo que tarda un modelo en ejecutar la inferencia en un hardware determinado. Cuanto menor sea el tiempo, más rápido será el modelo.

El rendimiento que necesita depende de su aplicación. El rendimiento puede ser importante para aplicaciones como el video en tiempo real, en las que puede ser importante analizar cada fotograma en el tiempo que transcurre antes de que se dibuje el siguiente (por ejemplo, la inferencia debe ser más rápida de 33 ms para realizar una inferencia en tiempo real en un flujo de video de 30 fps).

El rendimiento de los modelos MobileNet cuantizados con TensorFlow Lite oscila entre 3.7 ms y 80.3 ms.

Los números de referencia del rendimiento se generan con la <a href="https://www.tensorflow.org/lite/performance/benchmarks">herramienta de benchmarking</a>.

<table>
  <thead>
    <tr>
      <th>Nombre del modelo</th>
      <th>Tamaño del modelo</th>
      <th>Dispositivo</th>
      <th>NNAPI</th>
      <th>CPU</th>
    </tr>
  </thead>
  <tr>
    <td rowspan="3">       <a href="https://storage.googleapis.com/download.tensorflow.org/models/tflite/mobilenet_v1_1.0_224_quant_and_labels.zip">Mobilenet_V1_1.0_224_quant</a>
</td>
    <td rowspan="3">       4.3 Mb</td>
    <td>Pixel 3 (Android 10)</td>
    <td>6 ms</td>
    <td>13 ms*</td>
  </tr>
   <tr>
     <td>Pixel 4 (Android 10)</td>
    <td>3.3 ms</td>
    <td>5 ms*</td>
  </tr>
   <tr>
     <td>iPhone XS (iOS 12.4.1)</td>
     <td></td>
    <td>11 ms**</td>
  </tr>
</table>

* 4 hilos usados.

** 2 hilos usados en el iPhone para obtener el mejor resultado de rendimiento.

### Precisión del modelo

La precisión se mide en términos de la frecuencia con la que el modelo clasifica correctamente una imagen. Por ejemplo, cabe esperar que un modelo con una precisión declarada del 60 % clasifique correctamente una imagen un promedio del 60 % de las veces.

Las métricas de precisión más relevantes son Top-1 y Top-5. Top-1 se refiere a la frecuencia con la que la etiqueta correcta aparece como la etiqueta con mayor probabilidad en la salida del modelo. Top-5 se refiere a la frecuencia con la que la etiqueta correcta aparece entre las 5 probabilidades más altas en la salida del modelo.

El Top-5 de precisión de los modelos MobileNet cuantificados con TensorFlow Lite oscila entre el 64.4 y el 89.9 %.

### Tamaño del modelo

El tamaño en disco de un modelo varía en función de su rendimiento y precisión. El tamaño puede ser importante para el desarrollo móvil (donde puede influir en el tamaño de las descargas de la app) o cuando se trabaja con hardware (donde el almacenamiento disponible puede ser limitado).

El tamaño de los modelos MobileNet cuantizados por TensorFlow Lite oscila entre 0.5 y 3.4 MB.

## Lecturas y recursos complementarios

Use los siguientes recursos para aprender más sobre conceptos relacionados con la clasificación de imágenes:

- [Clasificación de imágenes usando TensorFlow](https://www.tensorflow.org/tutorials/images/classification)
- [Clasificación de imágenes con CNNs](https://www.tensorflow.org/tutorials/images/cnn)
- [Aprendizaje por transferencia](https://www.tensorflow.org/tutorials/images/transfer_learning)
- [Aumentación de datos](https://www.tensorflow.org/tutorials/images/data_augmentation)
