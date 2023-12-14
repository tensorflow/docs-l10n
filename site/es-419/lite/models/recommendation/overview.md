# Recomendación

<table class="tfo-notebook-buttons" align="left">   <td>     <a target="_blank" href="https://www.tensorflow.org/lite/examples/recommendation/overview"><img src="https://www.tensorflow.org/images/tf_logo_32px.png">Ver en TensorFlow.org</a>   </td>   {% dynamic if request.tld != 'cn' %}<td>     <a target="_blank" href="https://colab.research.google.com/github/tensorflow/examples/blob/master/lite/examples/recommendation/ml/ondevice_recommendation.ipynb"><img src="https://www.tensorflow.org/images/colab_logo_32px.png">Ejecutar en Google Colab</a>   </td>{% dynamic endif %}   <td>     <a target="_blank" href="https://github.com/tensorflow/examples/blob/master/lite/examples/recommendation/ml/ondevice_recommendation.ipynb"><img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png">Ver fuente en GitHub</a>   </td>
</table>

Las recomendaciones personalizadas se usan ampliamente para una gran variedad de casos de uso en dispositivos móviles, como la recuperación de contenidos multimedia, la sugerencia de productos de compra y la recomendación de la próxima app. Si está interesado en ofrecer recomendaciones personalizadas en su aplicación respetando la privacidad del usuario, le recomendamos que explore el siguiente ejemplo y conjunto de herramientas.

Nota: Para personalizar un modelo, pruebe [Model Maker de TensorFlow Lite](https://www.tensorflow.org/lite/guide/model_maker).

## Empecemos

<img src="images/screenshot.gif" class="attempt-right" style="max-width: 300px">

Proporcionamos una aplicación de muestra de TensorFlow Lite que demuestra cómo recomendar artículos relevantes a los usuarios en Android.

<a class="button button-primary" href="https://github.com/tensorflow/examples/tree/master/lite/examples/recommendation/android">Ejemplo en Android</a>

Si utiliza una plataforma distinta de Android o ya está familiarizado con las API de TensorFlow Lite, puede descargar nuestro modelo de recomendación para principiantes.

<a class="button button-primary" href="https://storage.googleapis.com/download.tensorflow.org/models/tflite/recommendation/20200720/recommendation.tar.gz">Descargue el modelo inicial</a>

También brindamos un script de entrenamiento en Github para entrenar su propio modelo de forma configurable.

<a class="button button-primary" href="https://github.com/tensorflow/examples/tree/master/lite/examples/recommendation/ml">Codigo de entrenamiento</a>

## Comprender la arquitectura del modelo

Aprovechamos una arquitectura de modelo de doble codificador, con codificador de contexto para codificar el historial secuencial del usuario y codificador de etiqueta para codificar el candidato de recomendación predicho. La similitud entre las codificaciones de contexto y etiqueta se usa para representar la probabilidad de que el candidato predicho satisfaga las necesidades del usuario.

Con esta base de código se proporcionan tres técnicas diferentes de codificación secuencial del historial del usuario:

- Codificador de bolsa de palabras (BOW): promedia las incrustaciones de las actividades de los usuarios sin tener en cuenta el orden del contexto.
- Codificador de redes neuronales convolucionales (CNN): aplicación de múltiples capas de redes neuronales convolucionales para generar una codificación contextual.
- Codificador de red neuronal recurrente (RNN): aplicación de la red neuronal recurrente para codificar la secuencia contextual.

Para modelar cada actividad del usuario, podríamos usar el ID del elemento de actividad (basado en el ID) , o múltiples características del elemento (basado en las características), o una combinación de ambos. El modelo basado en características utiliza múltiples características para codificar colectivamente el comportamiento de los usuarios. Con esta base de código, podría crear modelos basados en ID o en características de forma configurable.

Tras el entrenamiento, se exportará un modelo TensorFlow Lite que puede proporcionar directamente predicciones top-K entre los candidatos a recomendación.

## Use sus datos de entrenamiento

Además del modelo entrenado, ofrecemos un [toolkit de código abierto en GitHub](https://github.com/tensorflow/examples/tree/master/lite/examples/recommendation/ml) para entrenar modelos con sus propios datos. Puede seguir este tutorial para aprender a usar el kit de herramientas y a implementar los modelos entrenados en sus propias aplicaciones móviles.

Siga este [tutorial](https://github.com/tensorflow/examples/tree/master/lite/examples/recommendation/ml/ondevice_recommendation.ipynb) para aplicar la misma técnica utilizada aquí para entrenar un modelo de recomendación utilizando sus propios conjuntos de datos.

## Ejemplos

Como ejemplos, entrenamos modelos de recomendación con enfoques basados tanto en ID como en características. El modelo basado en ID sólo toma como entrada los ID de las películas, y el modelo basado en características toma como entrada tanto los ID de las películas como los ID de los géneros de las películas. Encuentre los siguientes ejemplos de entradas y salidas.

Entradas

- IDs de películas de contexto:

    - El Rey León (ID: 362)
    - Toy Story (ID: 1)
    - (y más)

- Contexto de las ID de género de la película:

    - Animación (ID: 15)
    - Infantil (ID: 9)
    - Musical (ID: 13)
    - Animación (ID: 15)
    - Infantil (ID: 9)
    - Comedia (ID: 2)
    - (y más)

Salidas

- IDs de películas recomendadas.
    - Toy Story 2 (ID: 3114)
    - (y más)

Nota: El modelo preentrenado se construye a partir del conjunto de datos [MovieLens](https://grouplens.org/datasets/movielens/1m/) con fines de investigación.

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
  <tbody>
    <tr>
      </tr>
<tr>
        <td rowspan="3">           <a href="https://storage.googleapis.com/download.tensorflow.org/models/tflite/recommendation/20200720/model.tar.gz">recomendación (ID de la película como entrada)</a>
</td>
        <td rowspan="3">           0.52 Mb</td>
        <td>Pixel 3</td>
        <td>0.09 ms*</td>
      </tr>
       <tr>
         <td>Pixel 4</td>
        <td>0.05 ms*</td>
      </tr>
    
    <tr>
      </tr>
<tr>
        <td rowspan="3">           <a href="https://storage.googleapis.com/download.tensorflow.org/models/tflite/recommendation/20210317/recommendation_cnn_i10i32o100.tflite">recomendación (ID de la película y género de la película como entradas)</a>
</td>
        <td rowspan="3">           1.3 Mb</td>
        <td>Pixel 3</td>
        <td>0.13 ms*</td>
      </tr>
       <tr>
         <td>Pixel 4</td>
        <td>0.06 ms*</td>
      </tr>
    
  </tbody>
</table>

* 4 hilos usados.

## Use sus datos de entrenamiento

Además del modelo entrenado, proporcionamos un [toolkit de código abierto en GitHub](https://github.com/tensorflow/examples/tree/master/lite/examples/recommendation/ml) para entrenar modelos con sus propios datos. Puede seguir este tutorial para aprender a usar el kit de herramientas y a implementar los modelos entrenados en sus propias aplicaciones móviles.

Siga este [tutorial](https://github.com/tensorflow/examples/tree/master/lite/examples/recommendation/ml/ondevice_recommendation.ipynb) para aplicar la misma técnica utilizada aquí para entrenar un modelo de recomendación utilizando sus propios conjuntos de datos.

## Consejos para la personalización del modelo con sus datos

El modelo preentrenado integrado en esta aplicación demo se ha entrenado con el conjunto de datos [MovieLens](https://grouplens.org/datasets/movielens/1m/), es posible que desee modificar la configuración del modelo en función de sus propios datos, como el tamaño del vocabulario, las dimensiones de incrustación y la longitud del contexto de entrada. He aquí algunos consejos:

- Longitud del contexto de entrada: La mejor longitud del contexto de entrada varía según los conjuntos de datos. Sugerimos seleccionar la longitud del contexto de entrada en función de cuánto se correlacionan los eventos de la etiqueta con los intereses a largo plazo frente al contexto a corto plazo.

- Selección del tipo de codificador: sugerimos elegir el tipo de codificador en función de la longitud del contexto de entrada. El codificador de bolsa de palabras funciona bien para longitudes de contexto de entrada cortas (por ejemplo, &lt;10), los codificadores CNN y RNN aportan más capacidad de resumen para longitudes de contexto de entrada largas.

- Usar características subyacentes para representar los elementos o las actividades del usuario podría mejorar el rendimiento del modelo, acomodar mejor los elementos nuevos, reducir posiblemente los espacios de incrustación y, por tanto, el consumo de memoria y ser más amigable con el contenido integrado en el dispositivo.
