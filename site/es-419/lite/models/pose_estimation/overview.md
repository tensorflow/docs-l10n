# Estimación de la pose

<img src="../images/pose.png" class="attempt-right">

La estimación de la pose es la tarea de usar un modelo ML para estimar la pose de una persona a partir de una imagen o un vídeo mediante la estimación de las ubicaciones espaciales de las articulaciones clave del cuerpo (puntos clave).

## Empecemos

Si es nuevo en TensorFlow Lite y trabaja con Android o iOS, explore las siguientes aplicaciones de ejemplo que pueden ayudarle a empezar.

<a class="button button-primary" href="https://github.com/tensorflow/examples/tree/master/lite/examples/pose_estimation/android"> Ejemplo de Android </a><a class="button button-primary" href="https://github.com/tensorflow/examples/tree/master/lite/examples/pose_estimation/ios"> Ejemplo de iOS</a>

Si está familiarizado con las [APIs TensorFlow Lite](https://www.tensorflow.org/api_docs/python/tf/lite), descargue el modelo de estimación de pose MoveNet de inicio y los archivos de apoyo.

<a class="button button-primary" href="https://tfhub.dev/s?q=movenet">Descargue el modelo inicial</a>

Si desea probar la estimación de pose en un navegador web, eche un vistazo a la <a href="https://storage.googleapis.com/tfjs-models/demos/pose-detection/index.html?model=movenet"> TensorFlow JS Demo</a>.

## Descripción del modelo

### Cómo funciona

La estimación de la pose se refiere a las técnicas de visión por computadora que detectan figuras humanas en imágenes y vídeos, de forma que se pueda determinar, por ejemplo, dónde aparece el codo de alguien en una imagen. Es importante ser consciente del hecho de que la estimación de la pose se limita a estimar dónde están las articulaciones clave del cuerpo y no reconoce quién aparece en una imagen o un vídeo.

Los modelos de estimación de la pose toman como entrada una imagen de cámara procesada y emiten información sobre los puntos clave. Los puntos clave detectados se indexan mediante un ID de pieza, con una puntuación de confianza entre 0.0 y 1.0. Las puntuaciones de confianza indican la probabilidad de que exista un punto clave en esa posición.

Proporcionamos la implementación de referencia de dos modelos de estimación de la pose de TensorFlow Lite:

- MoveNet: el modelo de estimación de poses más avanzado disponible en dos versiones: Lighting y Thunder. Vea una comparación entre estos dos en la sección siguiente.
- PoseNet: el modelo de estimación de la pose de la generación anterior lanzado en 2017.

Las distintas articulaciones del cuerpo detectadas por el modelo de estimación de la pose se tabulan a continuación:

<table style="width: 30%;">
  <thead>
    <tr>
      <th>Id</th>
      <th>Parte</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>nose</td>
    </tr>
    <tr>
      <td>1</td>
      <td>leftEye</td>
    </tr>
    <tr>
      <td>2</td>
      <td>rightEye</td>
    </tr>
    <tr>
      <td>3</td>
      <td>leftEar</td>
    </tr>
    <tr>
      <td>4</td>
      <td>rightEar</td>
    </tr>
    <tr>
      <td>5</td>
      <td>leftShoulder</td>
    </tr>
    <tr>
      <td>6</td>
      <td>rightShoulder</td>
    </tr>
    <tr>
      <td>7</td>
      <td>leftElbow</td>
    </tr>
    <tr>
      <td>8</td>
      <td>rightElbow</td>
    </tr>
    <tr>
      <td>9</td>
      <td>leftWrist</td>
    </tr>
    <tr>
      <td>10</td>
      <td>rightWrist</td>
    </tr>
    <tr>
      <td>11</td>
      <td>leftHip</td>
    </tr>
    <tr>
      <td>12</td>
      <td>rightHip</td>
    </tr>
    <tr>
      <td>13</td>
      <td>leftKnee</td>
    </tr>
    <tr>
      <td>14</td>
      <td>rightKnee</td>
    </tr>
    <tr>
      <td>15</td>
      <td>leftAnkle</td>
    </tr>
    <tr>
      <td>16</td>
      <td>rightAnkle</td>
    </tr>
  </tbody>
</table>

A continuación se muestra un ejemplo de salida:

<img src="https://storage.googleapis.com/download.tensorflow.org/example_images/movenet_demo.gif" alt="Animación que muestra la estimación de poses">

## Puntos de referencia del rendimiento

MoveNet está disponible en dos versiones:

- MoveNet.Lightning es más pequeño, más rápido pero menos preciso que la versión Thunder. Puede funcionar en tiempo real en los smartphones modernos.
- MoveNet.Thunder es la versión más precisa pero también más grande y lenta que Lightning. Es útil para los casos de uso que requieren una precisión más alta.

MoveNet supera a PoseNet en diversos conjuntos de datos, especialmente en las imágenes de acciones de fitness. Por lo tanto, recomendamos usar MoveNet en lugar de PoseNet.

Los números de referencia del rendimiento se generan con la herramienta [descrita aquí](../../performance/measurement). Las cifras de precisión (mAP) se miden en un subconjunto del [conjunto de datos COCO](https://cocodataset.org/#home) en el que filtramos y recortamos cada imagen para que contenga una sola persona .

<table>
<thead>
  <tr>
    <th rowspan="2">Modelo</th>
    <th rowspan="2">Tamaño (MB)</th>
    <th rowspan="2">mAP</th>
    <th colspan="3">Latencia (ms)</th>
  </tr>
  <tr>
    <td>Pixel 5 - 4 hilos de CPU</td>
    <td>Pixel 5 - GPU</td>
    <td>Raspberry Pi 4 - 4 hilos de CPU</td>
  </tr>
</thead>
<tbody>
  <tr>
    <td>       <a href="https://tfhub.dev/google/lite-model/movenet/singlepose/thunder/tflite/float16/4">MoveNet.Thunder (FP16 cuantizado)</a>
</td>
    <td>12.6 MB</td>
    <td>72.0</td>
    <td>155 ms</td>
    <td>45 ms</td>
    <td>594 ms</td>
  </tr>
  <tr>
    <td>       <a href="https://tfhub.dev/google/lite-model/movenet/singlepose/thunder/tflite/int8/4">MoveNet.Thunder (INT8 cuantizado)</a>
</td>
    <td>7.1 MB</td>
    <td>68.9</td>
    <td>100 ms</td>
    <td>52 ms</td>
    <td>251 ms</td>
  </tr>
  <tr>
    <td>       <a href="https://tfhub.dev/google/lite-model/movenet/singlepose/lightning/tflite/float16/4">MoveNet.Lightning (FP16 cuantizado)</a>
</td>
    <td>4.8 MB</td>
    <td>63.0</td>
    <td>60 ms</td>
    <td>25 ms</td>
    <td>186 ms</td>
  </tr>
  <tr>
    <td>       <a href="https://tfhub.dev/google/lite-model/movenet/singlepose/lightning/tflite/int8/4">MoveNet.Lightning (INT8 cuantizado)</a>
</td>
    <td>2.9 MB</td>
    <td>57.4</td>
    <td>52 ms</td>
    <td>28 ms</td>
    <td>95 ms</td>
  </tr>
  <tr>
    <td>       <a href="https://storage.googleapis.com/download.tensorflow.org/models/tflite/posenet_mobilenet_v1_100_257x257_multi_kpt_stripped.tflite">PoseNet(MobileNetV1 backbone, FP32)</a>
</td>
    <td>13.3 MB</td>
    <td>45.6</td>
    <td>80 ms</td>
    <td>40 ms</td>
    <td>338 ms</td>
  </tr>
</tbody>
</table>

## Lecturas y recursos complementarios

- Visite este [artículo de blog](https://blog.tensorflow.org/2021/08/pose-estimation-and-classification-on-edge-devices-with-MoveNet-and-TensorFlow-Lite.html) para saber más sobre la estimación de la pose usando MoveNet y TensorFlow Lite.
- Consulte este [artículo de blog](https://blog.tensorflow.org/2021/05/next-generation-pose-detection-with-movenet-and-tensorflowjs.html) para saber más sobre la estimación de la pose en la web.
- Vea este [tutorial](https://www.tensorflow.org/hub/tutorials/movenet) para aprender a ejecutar MoveNet en Python usando un modelo de TensorFlow Hub.
- Coral/EdgeTPU puede hacer que la estimación de la pose se ejecute mucho más rápido en los dispositivos periféricos. Consulte [EdgeTPU-optimized models](https://coral.ai/models/pose-estimation/) para más detalles.
- Lea el documento de PoseNet [aquí](https://arxiv.org/abs/1803.08225)

Además, eche un vistazo a estos casos de uso de la estimación de la pose.

<ul>
  <li><a href="https://vimeo.com/128375543">‘PomPom Mirror’</a></li>
  <li><a href="https://youtu.be/I5__9hq-yas">Una asombrosa instalación artística le convierte en un pájaro | Chris Milk "La traición del santuario"</a></li>
  <li><a href="https://vimeo.com/34824490">Puppet Parade - Marionetas interactivas de Kinect</a></li>
  <li><a href="https://vimeo.com/2892576">Messa di Voce (rendimiento), extractos</a></li>
  <li><a href="https://www.instagram.com/p/BbkKLiegrTR/">Realidad aumentada</a></li>
  <li><a href="https://www.instagram.com/p/Bg1EgOihgyh/">Animación interactiva</a></li>
  <li><a href="https://www.runnersneed.com/expert-advice/gear-guides/gait-analysis.html">Análisis de la marcha</a></li>
</ul>
