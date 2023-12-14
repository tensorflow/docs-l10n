# Clasificación de vídeos

<img src="../images/video.png" class="attempt-right">

La *clasificación de vídeos* es la tarea de aprendizaje automático que consiste en identificar lo que representa un vídeo. Un modelo de clasificación de vídeo se entrena en un conjunto de datos de vídeo que contiene un conjunto de clases únicas, como diferentes acciones o movimientos. El modelo recibe fotogramas de vídeo como entrada y emite la probabilidad de que cada clase esté representada en el vídeo.

Tanto los modelos de clasificación de vídeo como los de clasificación de imágenes usan imágenes como entradas para predecir las probabilidades de que esas imágenes pertenezcan a clases predefinidas. Sin embargo, un modelo de clasificación de vídeo también procesa las relaciones espaciotemporales entre fotogramas adyacentes para reconocer las acciones en un vídeo.

Por ejemplo, un modelo de *reconocimiento de acciones en vídeo* puede entrenarse para identificar acciones humanas como correr, aplaudir y saludar. La siguiente imagen muestra la salida de un modelo de clasificación de vídeo en Android.

<img src="https://storage.googleapis.com/download.tensorflow.org/models/tflite/screenshots/push-up-classification.gif" alt="Captura de pantalla de ejemplo de Android">

## Empecemos

Si utiliza una plataforma que no sea Android o Raspberry Pi, o si ya está familiarizado con las [API de TensorFlow Lite](https://www.tensorflow.org/api_docs/python/tf/lite), descargue el modelo de clasificación de vídeo de inicio y los archivos de soporte. También puede construir su propia canalización de inferencia personalizada usando la [Librería de soporte de TensorFlow Lite](../../inference_with_metadata/lite_support).

<a class="button button-primary" href="https://tfhub.dev/tensorflow/lite-model/movinet/a0/stream/kinetics-600/classification/tflite/int8/1">Descargar modelo inicial con metadatos</a>

Si no tiene mucha experiencia en TensorFlow Lite y trabaja con Android o Raspberry Pi, explore las siguientes aplicaciones de ejemplo que le ayudarán a empezar.

### Android

La aplicación Android usa la cámara trasera del dispositivo para la clasificación continua de vídeo. La inferencia se realiza usando la [API Java de TensorFlow Lite](https://www.tensorflow.org/lite/api_docs/java/org/tensorflow/lite/package-summary). La app demo clasifica fotogramas y muestra las clasificaciones predichas en tiempo real.

<a class="button button-primary" href="https://github.com/tensorflow/examples/tree/master/lite/examples/video_classification/android">Ejemplo en Android</a>

### Raspberry Pi

El ejemplo de la Raspberry Pi usa TensorFlow Lite con Python para realizar una clasificación de vídeo continua. Conecte la Raspberry Pi a una cámara, como la Pi Camera, para realizar la clasificación de vídeo en tiempo real. Para ver los resultados de la cámara, conecte un monitor a la Raspberry Pi y use SSH para acceder al shell de la Pi (para evitar conectar un teclado a la Pi).

Antes de empezar, [configure](https://projects.raspberrypi.org/en/projects/raspberry-pi-setting-up) su Raspberry Pi con Raspberry Pi OS (preferiblemente actualizado a Buster).

<a class="button button-primary" href="https://github.com/tensorflow/examples/tree/master/lite/examples/video_classification/raspberry_pi%20">Ejemplo de Raspberry Pi</a>

## Descripción del modelo

Las redes móviles de vídeo ([MoViNets](https://github.com/tensorflow/models/tree/master/official/projects/movinet)) son una familia de eficientes modelos de clasificación de vídeo optimizados para dispositivos móviles. Las MoViNets demuestran una precisión y eficiencia de vanguardia en varios conjuntos de datos de reconocimiento de acciones de vídeo a gran escala, lo que las hace muy adecuadas para tareas de *reconocimiento de acciones de vídeo*.

Existen tres variantes del modelo [MoviNet](https://tfhub.dev/s?deployment-format=lite&q=movinet) para TensorFlow Lite: [MoviNet-A0](https://tfhub.dev/tensorflow/movinet/a0/stream/kinetics-600/classification), [MoviNet-A1](https://tfhub.dev/tensorflow/movinet/a1/stream/kinetics-600/classification) y [MoviNet-A2](https://tfhub.dev/tensorflow/movinet/a2/stream/kinetics-600/classification). Estas variantes se entrenaron con el conjunto de datos [Kinetics-600](https://arxiv.org/abs/1808.01340) para reconocer 600 acciones humanas diferentes. *MoviNet-A0* es la más pequeña, rápida y menos precisa. *MoviNet-A2* es la más grande, lenta y precisa. *MoviNet-A1* es un término medio entre A0 y A2.

### Cómo funciona

Durante el entrenamiento, a un modelo de clasificación de vídeo se le facilitan vídeos y sus *etiquetas* asociadas. Cada etiqueta es el nombre de un concepto distinto, o clase, que el modelo aprenderá a reconocer. Para el *reconocimiento de acciones en vídeo*, los vídeos serán de acciones humanas y las etiquetas serán la acción asociada.

El modelo de clasificación de vídeo puede aprender a predecir si los nuevos vídeos pertenecen a alguna de las clases proporcionadas durante el entrenamiento. Este proceso se denomina *inferencia*. También puede usar el [aprendizaje por transferencia](https://colab.research.google.com/github/tensorflow/models/blob/master/official/projects/movinet/movinet_tutorial.ipynb) para identificar nuevas clases de vídeos usando un modelo preexistente.

El modelo es un modelo de flujo que recibe vídeo continuo y responde en tiempo real. A medida que el modelo recibe un flujo de vídeo, identifica si alguna de las clases del conjunto de datos de entrenamiento está representada en el vídeo. Para cada fotograma, el modelo devuelve estas clases, junto con la probabilidad de que el vídeo represente la clase. Un ejemplo de salida en un momento dado podría verse así:

<table style="width: 40%;">
  <thead>
    <tr>
      <th>Acción</th>
      <th>Probabilidad</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>baile de salón</td>
      <td>0.02</td>
    </tr>
    <tr>
      <td>enhebrar la aguja</td>
      <td>0.08</td>
    </tr>
    <tr>
      <td>mover los dedos</td>
      <td>0.23</td>
    </tr>
    <tr>
      <td style="background-color: #fcb66d;">Agitar la mano</td>
      <td style="background-color: #fcb66d;">0.67</td>
    </tr>
  </tbody>
</table>

Cada acción en la salida corresponde a una etiqueta en los datos de entrenamiento. La probabilidad denota la probabilidad de que la acción aparezca en el vídeo.

### Entradas del modelo

El modelo acepta como entrada un flujo de fotogramas de vídeo RGB. El tamaño del vídeo de entrada es flexible, pero lo ideal es que coincida con la resolución de entrenamiento del modelo y la velocidad de fotogramas:

- **MoviNet-A0**: 172 x 172 a 5 fps
- **MoviNet-A1**: 172 x 172 a 5 fps
- **MoviNet-A1**: 224 x 224 a 5 fps

Se espera que los vídeos de entrada tengan valores de color dentro de la gama de 0 y 1, siguiendo las [convenciones comunes de entrada de imágenes](https://www.tensorflow.org/hub/common_signatures/images#input).

Internamente, el modelo también analiza el contexto de cada fotograma usando la información recopilada en fotogramas anteriores. Esto se consigue tomando estados internos de la salida del modelo y retroalimentándolo para los fotogramas siguientes.

### Salidas del modelo

El modelo devuelve una serie de etiquetas y sus puntuaciones correspondientes. Las puntuaciones son valores logit que representan la predicción para cada clase. Estas puntuaciones pueden convertirse en probabilidades usando la función softmax (`tf.nn.softmax`).

```python
    exp_logits = np.exp(np.squeeze(logits, axis=0))
    probabilities = exp_logits / np.sum(exp_logits)
```

Internamente, la salida del modelo también incluye los estados internos del modelo y lo realimenta para los próximos fotogramas.

## Puntos de referencia del rendimiento

Los números de las pruebas de rendimiento se generan con la herramienta [benchmarking](https://www.tensorflow.org/lite/performance/measurement). MoviNets sólo admite CPU.

El rendimiento de un modelo se mide por la cantidad de tiempo que tarda en ejecutar la inferencia en un hardware determinado. Un tiempo menor implica un modelo más rápido. La precisión se mide por la frecuencia con la que el modelo clasifica correctamente una clase en un vídeo.

<table>
  <thead>
    <tr>
      <th>Nombre del modelo</th>
      <th>Tamaño</th>
      <th>Precisión *</th>
      <th>Dispositivo</th>
      <th>CPU **</th>
    </tr>
  </thead>
  <tr>
    <td rowspan="2"> MoviNet-A0 (Entero cuantizado)</td>
    <td rowspan="2">       3.1 MB</td>
    <td rowspan="2">65 %</td>
    <td>Pixel 4</td>
    <td>5 ms</td>
  </tr>
   <tr>
    <td>Pixel 3</td>
    <td>11 ms</td>
  </tr>
    <tr>
    <td rowspan="2"> MoviNet-A1 (Entero cuantizado)</td>
    <td rowspan="2">       4.5 MB</td>
    <td rowspan="2">70 %</td>
    <td>Pixel 4</td>
    <td>8 ms</td>
  </tr>
   <tr>
    <td>Pixel 3</td>
    <td>19 ms</td>
  </tr>
      <tr>
    <td rowspan="2"> MoviNet-A2 (Entero cuantizado)</td>
    <td rowspan="2">       5.1 MB</td>
    <td rowspan="2">72 %</td>
    <td>Pixel 4</td>
    <td>15 ms</td>
  </tr>
   <tr>
    <td>Pixel 3</td>
    <td>36 ms</td>
  </tr>
</table>

* Precisión máxima (Top-1) medida en el conjunto de datos [Kinetics-600](https://arxiv.org/abs/1808.01340).

** Latencia medida cuando se ejecuta en CPU con 1 hilo.

## Personalización de modelos

Los modelos preentrenados están entrenados para reconocer 600 acciones humanas del conjunto de datos [Kinetics-600](https://arxiv.org/abs/1808.01340). También puede usar el aprendizaje por transferencia para volver a entrenar un modelo que reconozca acciones humanas que no están en el conjunto original. Para ello, necesita un conjunto de vídeos de entrenamiento para cada una de las nuevas acciones que desee incorporar al modelo.

Para obtener más información sobre el ajuste de modelos en datos personalizados, consulte el repo [repositorio MoViNets](https://github.com/tensorflow/models/tree/master/official/projects/movinet) y el [tutorial de MoViNets](https://colab.research.google.com/github/tensorflow/models/blob/master/official/projects/movinet/movinet_tutorial.ipynb).

## Lecturas y recursos complementarios

Use los siguientes recursos para saber más sobre los conceptos analizados en esta página:

- [Repositorio de MoViNets](https://github.com/tensorflow/models/tree/master/official/projects/movinet)
- [Documento sobre MoViNets](https://arxiv.org/abs/2103.11511)
- [Modelos MoViNet preentrenados](https://tfhub.dev/s?deployment-format=lite&q=movinet)
- [Tutorial de MoViNets](https://colab.research.google.com/github/tensorflow/models/blob/master/official/projects/movinet/movinet_tutorial.ipynb)
- [Conjuntos de datos de Kinetics](https://deepmind.com/research/open-source/kinetics)
