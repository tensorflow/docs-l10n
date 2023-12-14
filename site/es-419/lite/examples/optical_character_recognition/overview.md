# Reconocimiento óptico de caracteres (OCR)

El reconocimiento óptico de caracteres (OCR) es el proceso de reconocer caracteres a partir de imágenes usando técnicas de visión por computadora y aprendizaje automático. Esta app de referencia muestra cómo usar TensorFlow Lite para hacer OCR. Usa una combinación de [modelo de detección de texto](https://tfhub.dev/sayakpaul/lite-model/east-text-detector/fp16/1) y un [modelo de reconocimiento de texto](https://tfhub.dev/tulasiram58827/lite-model/keras-ocr/float16/2) como canalización de OCR para reconocer caracteres de texto.

## Empecemos

<img src="images/screenshot.gif" class="attempt-right" style="max-width: 300px">

Si es nuevo en TensorFlow Lite y trabaja con Android, le recomendamos que explore las siguientes aplicaciones de ejemplo que pueden ayudarle a empezar.

<a class="button button-primary" href="https://github.com/tensorflow/examples/tree/master/lite/examples/optical_character_recognition/android">Ejemplo en Android</a>

Si utiliza una plataforma distinta de Android o ya está familiarizado con las [API de TensorFlow Lite](https://www.tensorflow.org/api_docs/python/tf/lite), puede descargar los modelos de [TF Hub](https://tfhub.dev/).

## Cómo funciona

Las tareas de OCR se suelen desglosar en 2 etapas. En primer lugar, usamos un modelo de detección de texto para detectar los cuadros delimitadores alrededor de posibles textos. En segundo lugar, introducimos los recuadros delimitadores procesados en un modelo de reconocimiento de texto para determinar los caracteres específicos dentro de los mismos (también tenemos que hacer supresión no maximal, transformación de perspectiva, etc. antes del reconocimiento de texto). En nuestro caso, ambos modelos proceden de TensorFlow Hub y son modelos cuantizados FP16.

## Puntos de referencia del rendimiento

Los números de referencia del rendimiento se generan con la herramienta [descrita aquí](https://www.tensorflow.org/lite/performance/benchmarks).

<table>
  <thead>
    <tr>
      <th>Nombre del modelo</th>
      <th>Tamaño del modelo</th>
      <th>Dispositivo</th>
      <th>CPU</th>
      <th>GPU</th>
    </tr>
  </thead>
  <tr>
    <td>       <a href="https://tfhub.dev/sayakpaul/lite-model/east-text-detector/fp16/1">Detección de texto</a>
</td>
    <td>45.9 Mb</td>
     <td>Pixel 4 (Android 10)</td>
     <td>181.93 ms*</td>
     <td>89.77 ms*</td>
  </tr>
  <tr>
    <td>       <a href="https://tfhub.dev/tulasiram58827/lite-model/keras-ocr/float16/2">Reconocimiento de texto</a>
</td>
    <td>16.8 Mb</td>
     <td>Pixel 4 (Android 10)</td>
     <td>338.33 ms*</td>
     <td>N/A**</td>
  </tr>
</table>

* 4 hilos usados.

** este modelo no podría usar el delegado de GPU ya que necesitamos ops de TensorFlow para ejecutarlo

## Entradas

El modelo de detección de texto acepta un Tensor 4-D `float32` de (1, 320, 320, 3) como entrada.

El modelo de reconocimiento de texto acepta un Tensor 4-D `float32` de (1, 31, 200, 1) como entrada.

## Salidas

El modelo de detección de texto devuelve un Tensor 4-D `float32` de forma (1, 80, 80, 5) como cuadro delimitador y un tensor 4-D `float32` de forma (1, 80, 80, 5) como puntuación de detección.

El modelo de reconocimiento de texto devuelve un Tensor 2-D `float32` de forma (1, 48) como los índices de mapeo a la lista del alfabeto '0123456789abcdefghijklmnopqrstuvwxyz'.

## Limitaciones

- El actual [modelo de reconocimiento de texto](https://tfhub.dev/tulasiram58827/lite-model/keras-ocr/float16/2) está entrenado usando datos sintéticos con letras y números en inglés, por lo que sólo se admite el inglés.

- Los modelos no son lo suficientemente generales para el OCR en el mundo real (digamos, imágenes aleatorias tomadas por la cámara de un smartphone en condiciones de poca luz).

Así que hemos seleccionado 3 logotipos de productos de Google sólo para demostrar cómo hacer OCR con TensorFlow Lite. Si está buscando un producto de OCR de grado de producción listo para usar, debería considerar [Google ML Kit](https://developers.google.com/ml-kit/vision/text-recognition). ML Kit, que usa TFLite en su base, debería ser suficiente para la mayoría de los casos de uso de OCR, pero hay algunos casos en los que puede querer generar su propia solución de OCR con TFLite. Algunos ejemplos son:

- Tiene sus propios modelos TFLite de detección/reconocimiento de texto que le gustaría usar
- Tiene requisitos empresariales especiales (por ejemplo, reconocer textos que están al revés) y necesita personalizar la canalización de OCR
- Desea admitir idiomas no cubiertos por el ML Kit
- Los dispositivos de sus usuarios objetivo no tienen necesariamente instalados los servicios de Google Play

## Referencias

- Ejemplo de detección/reconocimiento de texto OpenCV: https://github.com/opencv/opencv/blob/master/samples/dnn/text_detection.cpp
- Proyecto comunitario OCR TFLite por colaboradores de la comunidad: https://github.com/tulasiram58827/ocr_tflite
- Detección de texto OpenCV: https://www.pyimagesearch.com/2018/08/20/opencv-text-detection-east-text-detector/
- Detección de texto basada en aprendizaje profundo usando OpenCV: https://learnopencv.com/deep-learning-based-text-detection-using-opencv-c-python/
