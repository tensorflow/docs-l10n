# Detección de objetos

Dada una imagen o un flujo de video, un modelo de detección de objetos puede identificar cuáles de un conjunto conocido de objetos podrían estar presentes y proporcionar información sobre sus posiciones dentro de la imagen.

Por ejemplo, esta captura de pantalla de la aplicación de ejemplo muestra cómo se han reconocido dos objetos y anotado sus posiciones:

<img src="images/android_apple_banana.png" width="30%" alt="Captura de pantalla de ejemplo de Android">

Nota: (1) Para integrar un modelo existente, pruebe la [Librería de tareas de TensorFlow Lite](../../inference_with_metadata/task_library/object_detector). (2) Para personalizar un modelo, pruebe [Model Maker de TensorFlow Lite](https://www.tensorflow.org/lite/guide/model_maker).

## Empecemos

Para aprender a usar la detección de objetos en una app móvil, explore las <a href="#example_applications_and_guides">Aplicaciones de ejemplo y guías</a>.

Si utiliza una plataforma distinta de Android o iOS, o si ya está familiarizado con las <a href="https://www.tensorflow.org/api_docs/python/tf/lite">APIs de TensorFlow Lite</a>, puede descargar nuestro modelo de detección de objetos inicial y las etiquetas que lo acompañan.

<a class="button button-primary" href="https://tfhub.dev/tensorflow/lite-model/ssd_mobilenet_v1/1/metadata/1?lite-format=tflite">Descargar modelo inicial con metadatos</a>

Para obtener más información sobre los metadatos y los campos asociados (p. ej.: `labels.txt`), consulte <a href="../../models/convert/metadata#read_the_metadata_from_models">Leer los metadatos de los modelos</a>.

Si desea entrenar un modelo de detección personalizado para su propia tarea, consulte <a href="#model-customization">Personalización del modelo</a>.

Para los siguientes casos de uso, deberá usar un tipo de modelo diferente:

<ul>
  <li>Predecir qué etiqueta única representa con mayor probabilidad la imagen (consulte la <a href="../image_classification/overview.md">clasificación de imágenes</a>)</li>
  <li>Predecir la composición de una imagen, por ejemplo el sujeto frente al fondo (consulte la <a href="../segmentation/overview.md">segmentación</a>)</li>
</ul>

### Ejemplos de aplicaciones y guías

Si es nuevo en TensorFlow Lite y trabaja con Android o iOS, le recomendamos explorar las siguientes aplicaciones de ejemplo que pueden ayudarle a empezar.

#### Android

Puede aprovechar la API sin necesidad de instalación de la [Librería de tareas de TensorFlow Lite](../../inference_with_metadata/task_library/object_detector) para integrar modelos de detección de objetos en tan solo unas líneas de código. También puede construir su propia canalización de inferencia personalizada usando la [API Java del intérprete de TensorFlow Lite](../../guide/inference#load_and_run_a_model_in_java).

El siguiente ejemplo de Android muestra la implementación de ambos métodos usando [la librería de tareas](https://github.com/tensorflow/examples/tree/master/lite/examples/object_detection/android_play_services) y [la API del intérprete](https://github.com/tensorflow/examples/tree/eb925e460f761f5ed643d17f0c449e040ac2ac45/lite/examples/object_detection/android/lib_interpreter), respectivamente.

<a class="button button-primary" href="https://github.com/tensorflow/examples/tree/master/lite/examples/object_detection/android">Ver ejemplo en Android</a>

#### iOS

Puede integrar el modelo usando la [API Swift de intérprete de TensorFlow Lite](../../guide/inference#load_and_run_a_model_in_swift). Vea el ejemplo de iOS a continuación.

<a class="button button-primary" href="https://github.com/tensorflow/examples/tree/master/lite/examples/object_detection/ios">Ver ejemplo en iOS</a>

## Descripción del modelo

Esta sección describe la firma de los modelos [Single-Shot Detector](https://arxiv.org/abs/1512.02325) convertidos a TensorFlow Lite desde la API [TensorFlow Object Detection](https://github.com/tensorflow/models/blob/master/research/object_detection/).

Un modelo de detección de objetos se entrena para detectar la presencia y la ubicación de varias clases de objetos. Por ejemplo, un modelo puede entrenarse con imágenes que contengan varias piezas de fruta, junto con una *etiqueta* que especifique la clase de fruta que representan (por ejemplo, una manzana, un plátano o una fresa), y datos que especifiquen dónde aparece cada objeto en la imagen.

Cuando posteriormente se proporciona una imagen al modelo, éste emitirá una lista de los objetos que detecta, la ubicación de un cuadro delimitador que contiene cada objeto y una puntuación que indica la confianza en que la detección fue correcta.

### Firma de entrada

El modelo toma una imagen como entrada.

Supongamos que la imagen esperada es de 300 x 300 pixeles, con tres canales (rojo, azul y verde) por pixel. Esto debería introducirse en el modelo como un búfer aplanado de 270,000 valores de bytes (300 x 300 x 3). Si el modelo está <a href="../../performance/post_training_quantization.md">cuantizado</a>, cada valor debe ser un único byte que represente un valor entre 0 y 255.

Puede echar un vistazo al código de nuestro [ejemplo de app](https://github.com/tensorflow/examples/tree/master/lite/examples/object_detection/android) para entender cómo realizar este preprocesamiento en Android.

### Firma de salida

El modelo produce cuatro arreglos, mapeados en los índices 0-4. Los arreglos 0, 1 y 2 describen `N` objetos detectados, correspondiendo un elemento de cada arreglo a cada objeto.

<table>
  <thead>
    <tr>
      <th>Índice</th>
      <th>Nombre</th>
      <th>Descripción</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>Ubicaciones</td>
      <td>Arreglo multidimensional de [N][4] valores de coma flotante entre 0 y 1, los arreglos interiores representan cajas delimitadoras de la forma [arriba, izquierda, abajo, derecha].</td>
    </tr>
    <tr>
      <td>1</td>
      <td>Clases</td>
      <td>Arreglo de N enteros (salida como valores de coma flotante) cada uno de los cuales indica el índice de una etiqueta de clase del archivo de etiquetas</td>
    </tr>
    <tr>
      <td>2</td>
      <td>Puntuaciones</td>
      <td>Arreglo de N valores de coma flotante entre 0 y 1 que representan la probabilidad de que se haya detectado una clase</td>
    </tr>
    <tr>
      <td>3</td>
      <td>Número de detecciones</td>
      <td>Valor entero de N</td>
    </tr>
  </tbody>
</table>

NOTA: El número de resultados (10 en el caso anterior) es un parámetro establecido al exportar el modelo de detección a TensorFlow Lite. Vea <a href="#model-customization">Personalización del modelo</a> para más detalles.

Por ejemplo, imagine que un modelo ha sido entrenado para detectar manzanas, plátanos y fresas. Cuando se le proporcione una imagen, emitirá un número establecido de resultados de detección: en este ejemplo, 5.

<table style="width: 60%;">
  <thead>
    <tr>
      <th>Clase</th>
      <th>Puntuación</th>
      <th>Ubicación</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Manzana</td>
      <td>0.92</td>
      <td>[18, 21, 57, 63]</td>
    </tr>
    <tr>
      <td>Plátano</td>
      <td>0.88</td>
      <td>[100, 30, 180, 150]</td>
    </tr>
    <tr>
      <td>Fresa</td>
      <td>0.87</td>
      <td>[7, 82, 89, 163]</td>
    </tr>
    <tr>
      <td>Plátano</td>
      <td>0.23</td>
      <td>[42, 66, 57, 83]</td>
    </tr>
    <tr>
      <td>Manzana</td>
      <td>0.11</td>
      <td>[6, 42, 31, 58]</td>
    </tr>
  </tbody>
</table>

#### Puntuación de confianza

Para interpretar estos resultados, podemos fijarnos en las puntuaciones y la ubicación de cada objeto detectado. Las puntuaciones son un número entre 0 y 1 que indica la confianza en que el objeto fue realmente detectado. Cuanto más se acerque el número a 1, mayor será la confianza del modelo.

Dependiendo de su aplicación, puede decidir un umbral de corte por debajo del cual descartará los resultados de la detección. Para el ejemplo actual, un punto de corte sensato es una puntuaciones de 0.5 (lo que significa una probabilidad del 50 % de que la detección sea válida). En ese caso, los dos últimos objetos del arreglo se ignorarían porque esas puntuaciones de confianza están por debajo de 0.5:

<table style="width: 60%;">
  <thead>
    <tr>
      <th>Clase</th>
      <th>Puntuación</th>
      <th>Ubicación</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Manzana</td>
      <td>0.92</td>
      <td>[18, 21, 57, 63]</td>
    </tr>
    <tr>
      <td>Plátano</td>
      <td>0.88</td>
      <td>[100, 30, 180, 150]</td>
    </tr>
    <tr>
      <td>Fresa</td>
      <td>0.87</td>
      <td>[7, 82, 89, 163]</td>
    </tr>
    <tr>
      <td style="background-color: #e9cecc; text-decoration-line: line-through;">Plátano</td>
      <td style="background-color: #e9cecc; text-decoration-line: line-through;">0.23</td>
      <td style="background-color: #e9cecc; text-decoration-line: line-through;">[42, 66, 57, 83]</td>
    </tr>
    <tr>
      <td style="background-color: #e9cecc; text-decoration-line: line-through;">Manzana</td>
      <td style="background-color: #e9cecc; text-decoration-line: line-through;">0.11</td>
      <td style="background-color: #e9cecc; text-decoration-line: line-through;">[6, 42, 31, 58]</td>
    </tr>
  </tbody>
</table>

El punto de corte que use debe basarse en si se siente más cómodo con los falsos positivos (objetos que se identifican erróneamente, o zonas de la imagen que se identifican erróneamente como objetos cuando no lo son), o con los falsos negativos (objetos auténticos que se pasan por alto porque su confianza era baja).

Por ejemplo, en la siguiente imagen, una pera (que no es un objeto que el modelo estuviera entrenado para detectar) fue identificada erróneamente como una "persona". Este es un ejemplo de falso positivo que podría ignorarse seleccionando un punto de corte adecuado. En este caso, un punto de corte de 0.6 (o 60 %) excluiría cómodamente el falso positivo.

<img src="images/false_positive.png" width="30%" alt="Captura de pantalla de ejemplo de Android que muestra un falso positivo">

#### Ubicación

Para cada objeto detectado, el modelo devolverá un arreglo de cuatro números que representan un rectángulo delimitador que rodea su posición. Para el modelo de inicio proporcionado, los números están ordenados de la siguiente manera:

<table style="width: 50%; margin: 0 auto;">
  <tbody>
    <tr style="border-top: none;">
      <td>[</td>
      <td>superior,</td>
      <td>izquierda,</td>
      <td>inferior,</td>
      <td>derecha</td>
      <td>]</td>
    </tr>
  </tbody>
</table>

El valor superior representa la distancia del borde superior del rectángulo desde la parte superior de la imagen, en pixeles. El valor izquierdo representa la distancia del borde izquierdo desde la izquierda de la imagen de entrada. Los otros valores representan los bordes inferior y derecho de forma similar.

Nota: Los modelos de detección de objetos aceptan imágenes de entrada de un tamaño específico. Es probable que sea diferente del tamaño de la imagen en bruto capturada por la cámara de su dispositivo, y tendrá que escribir código para recortar y escalar su imagen en bruto para que se ajuste al tamaño de entrada del modelo (hay ejemplos de esto en nuestras <a href="#get_started">aplicaciones de ejemplo</a>).<br><br>Los valores de pixel de salida del modelo se refieren a la posición en la imagen recortada y escalada, por lo que debe escalarlos para que se ajusten a la imagen en bruto para poder interpretarlos correctamente.

## Puntos de referencia del rendimiento

Los números de referencia de rendimiento para nuestro <a class="button button-primary" href="https://tfhub.dev/tensorflow/lite-model/ssd_mobilenet_v1/1/metadata/1?lite-format=tflite">modelo de inicio</a> se generan con la herramienta [descrita aquí](https://www.tensorflow.org/lite/performance/benchmarks).

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
    <td rowspan="3">       <a href="https://tfhub.dev/tensorflow/lite-model/ssd_mobilenet_v1/1/metadata/1?lite-format=tflite">COCO SSD MobileNet v1</a>
</td>
    <td rowspan="3">       27 Mb</td>
    <td>Pixel 3 (Android 10)</td>
    <td>22 ms</td>
    <td>46 ms*</td>
  </tr>
   <tr>
     <td>Pixel 4 (Android 10)</td>
    <td>20 ms</td>
    <td>29 ms*</td>
  </tr>
   <tr>
     <td>iPhone XS (iOS 12.4.1)</td>
     <td>7.6 ms</td>
    <td>11 ms**</td>
  </tr>
</table>

* 4 hilos usados.

** 2 hilos usados en el iPhone para obtener el mejor resultado de rendimiento.

## Personalización de modelos

### Modelo previamente entrenado

En el [Zoo de detección](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md#mobile-models) se pueden encontrar modelos de detección optimizados para móviles con diversas características de latencia y precisión. Cada uno de ellos sigue las firmas de entrada y salida descritas en las secciones siguientes.

La mayoría de los archivos zips de descarga contienen un archivo `model.tflite`. Si no hay ninguno, se puede generar un flatbuffer TensorFlow Lite usando [estas instrucciones](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/running_on_mobile_tensorflowlite.md). Los modelos SSD del zoo de detección de objetos [TF2](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md) también pueden convertirse a TensorFlow Lite usando [estas instrucciones](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/running_on_mobile_tf2.md). Es importante tener en cuenta que los modelos de detección no pueden convertirse directamente usando el [Convertidor de TensorFlow Lite](../../models/convert), ya que requieren un paso intermedio de generación de un modelo fuente apto para móviles. Los scripts enlazados anteriormente realizan este paso.

Tanto los scripts de exportación [TF1](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/running_on_mobile_tensorflowlite.md) como [TF2](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/running_on_mobile_tf2.md) tienen parámetros que pueden permitir un mayor número de objetos de salida o un postprocesado más lento y preciso. Por favor, use `--help` con los scripts para ver una lista exhaustiva de los argumentos soportados.

> Actualmente, la inferencia en el dispositivo sólo está optimizada con modelos SSD. Se está investigando un mejor soporte para otras arquitecturas como CenterNet y EfficientDet.

### ¿Cómo seleccionar un modelo para personalizarlo?

Cada modelo tiene sus propias características de precisión (cuantificada por el valor mAP) y latencia. Debe seleccionar el modelo que mejor se adapte a su caso de uso y al hardware previsto. Por ejemplo, los modelos [Edge TPU](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md#pixel4-edge-tpu-models) son ideales para la inferencia en la TPU Edge de Google en el Pixel 4.xz

Puede usar nuestra herramienta de [benchmark](https://www.tensorflow.org/lite/performance/measurement) para evaluar los modelos y seleccionar la opción más eficiente disponible.

## Ajuste de modelos sobre datos personalizados

Los modelos preentrenados que ofrecemos están entrenados para detectar 90 clases de objetos. Para ver la lista completa de clases, consulte el archivo de etiquetas en los <a href="https://tfhub.dev/tensorflow/lite-model/ssd_mobilenet_v1/1/metadata/1?lite-format=tflite">metadatos del modelo</a>.

Puede usar una técnica conocida como aprendizaje por transferencia para volver a entrenar un modelo para que reconozca clases que no están en el conjunto original. Por ejemplo, podría volver a entrenar el modelo para detectar múltiples tipos de vegetales, a pesar de que sólo hay un vegetal en los datos de entrenamiento originales. Para ello, necesitará un conjunto de imágenes de entrenamiento para cada una de las nuevas etiquetas que desee entrenar. Lo recomendable es usar la librería [Model Maker de TensorFlow Lite](https://www.tensorflow.org/lite/guide/model_maker), que simplifica el proceso de entrenamiento de un modelo TensorFlow Lite utilizando un conjunto de datos personalizado, con unas pocas líneas de código. Usa el aprendizaje por transferencia para reducir la cantidad de datos de entrenamiento necesarios y el tiempo. También puede aprender del [Colaboratorio de detección de pocas tomas](https://github.com/tensorflow/models/blob/master/research/object_detection/colab_tutorials/eager_few_shot_od_training_tflite.ipynb) como ejemplo de ajuste fino de un modelo preentrenado con pocos ejemplos.

Para afinar con conjuntos de datos más grandes, eche un vistazo a estas guías para entrenar sus propios modelos con la API de detección de objetos de TensorFlow: [TF1](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_training_and_evaluation.md), [TF2](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_training_and_evaluation.md). Una vez entrenados, se pueden convertir a un formato compatible con TFLite con las instrucciones aquí: [TF1](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/running_on_mobile_tensorflowlite.md), [TF2](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/running_on_mobile_tf2.md)
