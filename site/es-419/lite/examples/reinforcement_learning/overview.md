# Aprendizaje por refuerzo

Juegue a un juego de mesa contra un agente entrenado mediante aprendizaje por refuerzo e implementado con TensorFlow Lite.

## Empecemos

<img src="images/screenshot.gif" class="attempt-right" style="max-width: 300px">

Si es nuevo en TensorFlow Lite y trabaja con Android, le recomendamos que explore las siguientes aplicaciones de ejemplo que pueden ayudarle a empezar.

<a class="button button-primary" href="https://github.com/tensorflow/examples/tree/master/lite/examples/reinforcement_learning/android">Ejemplo en Android</a>

Si utiliza una plataforma distinta de Android o ya está familiarizado con las [APIs de TensorFlow Lite](https://www.tensorflow.org/api_docs/python/tf/lite)[, puede descargar nuestro modelo entrenado.](https://www.tensorflow.org/api_docs/python/tf/lite)

<a class="button button-primary" href="https://github.com/tensorflow/examples/blob/master/lite/examples/reinforcement_learning/android/app/src/main/assets/planestrike_tf.tflite">Descargue el modelo</a>

## Cómo funciona

El modelo está construido para que un agente juegue a un pequeño juego de mesa llamado 'Plane Strike'. Para ver una introducción rápida a este juego y sus reglas, consulte este archivo [LEEME](https://github.com/tensorflow/examples/tree/master/lite/examples/reinforcement_learning/android).

Subyacente a la interfaz de usuario de la app, hemos construido un agente que juega contra el jugador humano. El agente es un MLP de 3 capas que toma el estado del tablero como entrada y emite la puntuación prevista para cada una de las 64 casillas posibles del tablero. El modelo se entrena usando gradiente de política (REINFORCE) y puede encontrar el código de entrenamiento [aquí](https://github.com/tensorflow/examples/blob/master/lite/examples/reinforcement_learning/ml). Tras el entrenamiento del agente, convertimos el modelo en TFLite y lo implementamos en la app de Android.

Durante el juego real en la app de Android, cuando es el turno de actuar del agente, éste observa el estado del tablero del jugador humano (el tablero de la parte inferior), que contiene información sobre los impactos anteriores exitosos y fallidos (aciertos y errores), y usa el modelo entrenado para predecir dónde atacar a continuación, de modo que pueda terminar la partida antes que el jugador humano.

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
    <td rowspan="2">       <a href="https://github.com/tensorflow/examples/blob/master/lite/examples/reinforcement_learning/android/app/src/main/assets/planestrike.tflite">Gradiente de política</a>
</td>
    <td rowspan="2">       84 Kb</td>
    <td>Pixel 3 (Android 10)</td>
    <td>0.01 ms*</td>
  </tr>
   <tr>
     <td>Pixel 4 (Android 10)</td>
    <td>0.01 ms*</td>
  </tr>
</table>

* 1 hilo usado

## Entradas

El modelo acepta un Tensor 3-D `float32` de (1, 8, 8) como estado del tablero.

## Salidas

El modelo devuelve un Tensor 2-D `float32` de forma (1, 64) como las puntuaciones previstas para cada una de las 64 posiciones de golpe posibles.

## Entrene su propio modelo

Podría entrenar su propio modelo para un tablero más grande/pequeño cambiando el parámetro `BOARD_SIZE` en el [código de entrenamiento](https://github.com/tensorflow/examples/blob/master/lite/examples/reinforcement_learning/ml).
