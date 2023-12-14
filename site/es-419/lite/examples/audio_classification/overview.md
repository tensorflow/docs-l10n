# Clasificación del audio

<img src="../images/audio.png" class="attempt-right">

La tarea de identificar lo que representa un audio se denomina *clasificación de audio*. Un modelo de clasificación de audio se entrena para reconocer varios eventos de audio. Por ejemplo, puede entrenar un modelo para reconocer eventos que representan tres eventos diferentes: aplaudir, chasquear los dedos y teclear. TensorFlow Lite proporciona modelos preentrenados optimizados que puede implementar en sus aplicaciones Móviles. Obtenga más información sobre la clasificación de audio usando TensorFlow [aquí](https://www.tensorflow.org/tutorials/audio/simple_audio).

La siguiente imagen muestra la salida del modelo de clasificación de audio en Android.

<img src="images/android_audio_classification.png" width="30%" alt="Captura de pantalla de ejemplo de Android">

Nota: (1) Para integrar un modelo existente, pruebe la [Librería de tareas de TensorFlow Lite](https://www.tensorflow.org/lite/inference_with_metadata/task_library/audio_classifier). (2) Para personalizar un modelo, pruebe [Model Maker de TensorFlow Lite](https://www.tensorflow.org/lite/models/modify/model_maker/audio_classification).

## Empecemos

Si es nuevo en TensorFlow Lite y trabaja con Android, le recomendamos que explore las siguientes aplicaciones de ejemplo que pueden ayudarle a empezar.

Puede aprovechar la API lista para usar de la [Librería de tareas TensorFlow Lite](../../inference_with_metadata/task_library/audio_classifier) para integrar modelos de clasificación de audio en tan solo unas líneas de código. También puede construir su propia canalización de inferencia personalizada usando la [Librería de soporte de TensorFlow Lite](../../inference_with_metadata/lite_support).

El siguiente ejemplo de Android muestra la implementación usando la [librería de tareas de TFLite](https://github.com/tensorflow/examples/tree/master/lite/examples/audio_classification/android).

<a class="button button-primary" href="https://github.com/tensorflow/examples/tree/master/lite/examples/audio_classification/android">Ver ejemplo en Android</a>

<a class="button button-primary" href="https://github.com/tensorflow/examples/tree/master/lite/examples/sound_classification/ios">Ver ejemplo en iOS</a>

Si utiliza una plataforma distinta de Android/iOS, o si ya está familiarizado con las API de [TensorFlow Lite](https://www.tensorflow.org/api_docs/python/tf/lite), descargue el modelo de inicio y los archivos de soporte (si procede).

<a class="button button-primary" href="https://tfhub.dev/google/lite-model/yamnet/classification/tflite/1?lite-format=tflite">Descargue el modelo inicial de TensorFlow Hub</a>

## Descripción del modelo

YAMNet es un clasificador de eventos de audio que recibe la forma de onda de audio como entrada y realiza predicciones independientes para cada uno de los 521 eventos de audio de la antología [AudioSet](https://g.co/audioset). El modelo usa la arquitectura MobileNet v1 y fue entrenado usando el corpus AudioSet. Este modelo se publicó originalmente en el TensorFlow Model Garden, donde se encuentra el código fuente del modelo, el punto de verificación original del modelo y documentación más detallada.

### Cómo funciona

Existen dos versiones del modelo YAMNet convertido a TFLite:

- [YAMNet](https://tfhub.dev/google/yamnet/1) Es el modelo de clasificación de audio original, con un tamaño de entrada dinámico, adecuado para la implementación de Transfer Learning, Web y Mobile. También tiene una salida más compleja.

- [YAMNet/classification](https://tfhub.dev/google/lite-model/yamnet/classification/tflite/1) es una versión cuantizada con una trama de entrada de longitud fija más simple (15600 muestras) y devuelve un único vector de puntuaciones para 521 clases de eventos de audio.

### Entradas

El modelo acepta un Tensor 1-D `float32` o arreglo NumPy de longitud 15600 que contiene una forma de onda de 0.975 segundos representada como muestreos mono de 16 kHz en el rango `[-1.0, +1.0]`.

### Salidas

El modelo devuelve un Tensor 2-D `float32` de forma (1, 521) que contiene las puntuaciones previstas para cada una de las 521 clases de la ontología AudioSet que admite YAMNet. El índice de columna (0-520) del tensor de puntuaciones se mapea con el nombre de la clase AudioSet correspondiente utilizando el mapa de clases de YAMNet, que está disponible como archivo asociado `yamnet_label_list.txt` empaquetado en el archivo del modelo. Véase más abajo para su uso.

### Usos adecuados

Se puede usar YAMNet

- como clasificador independiente de eventos de audio que ofrece una línea de referencia razonable en una amplia variedad de eventos de audio.
- como un extractor de características de alto nivel: el resultado de la incrustación 1024-D de YAMNet puede usarse como características de entrada de otro modelo que, a continuación, puede entrenarse con una pequeña cantidad de datos para una tarea concreta. Esto permite crear rápidamente clasificadores de audio especializados sin necesidad de muchos datos etiquetados y sin tener que entrenar un gran modelo de extremo a extremo.
- como un arranque en caliente: los parámetros del modelo YAMNet pueden usarse para inicializar parte de un modelo mayor que permita un ajuste fino y una exploración del modelo más rápidos.

### Limitaciones

- Las salidas del clasificador de YAMNet no han sido calibradas entre clases, por lo que no puede tratar directamente las salidas como probabilidades. Para cualquier tarea determinada, es muy probable que necesite realizar una calibración con datos específicos de la tarea que le permitan asignar umbrales de puntuaciones y escalados adecuados por clase.
- YAMNet ha sido entrenado con millones de vídeos de YouTube y, aunque éstos son muy diversos, aún puede haber un desajuste de dominio entre el vídeo promedio de YouTube y las entradas de audio esperadas para cualquier tarea dada. Es de esperar que tenga que realizar algún tipo de ajuste y calibración para que YAMNet sea utilizable en cualquier sistema que construya.

## Personalización de modelos

Los modelos preentrenados suministrados están entrenados para detectar 521 clases de audio diferentes. Para obtener una lista completa de clases, consulte el archivo de etiquetas en el <a href="https://github.com/tensorflow/models/blob/master/research/audioset/yamnet/yamnet_class_map.csv">repositorio de modelos</a>.

Puede usar una técnica conocida como aprendizaje por transferencia para volver a entrenar un modelo para que reconozca clases que no están en el conjunto original. Por ejemplo, podría volver a entrenar el modelo para detectar múltiples cantos de pájaros. Para ello, necesitará un conjunto de audios de entrenamiento para cada una de las nuevas etiquetas que desee entrenar. Lo recomendable es usar la librería [Model Maker de TensorFlow Lite](https://www.tensorflow.org/lite/models/modify/model_maker/audio_classification), que simplifica el proceso de entrenamiento de un modelo TensorFlow Lite usando un conjunto de datos personalizado, en unas pocas líneas de código. Utiliza el aprendizaje por transferencia para reducir la cantidad de datos de entrenamiento necesarios y el tiempo. También puede aprender de [Aprendizaje por transferencia para el reconocimiento de audio](https://www.tensorflow.org/tutorials/audio/transfer_learning_audio) como ejemplo de aprendizaje por transferencia.

## Lecturas y recursos complementarios

Use los siguientes recursos para aprender más sobre conceptos relacionados con la clasificación del audio:

- [Clasificación de audio usando TensorFlow](https://www.tensorflow.org/tutorials/audio/simple_audio)
- [Aprendizaje por transferencia para el reconocimiento de audio](https://www.tensorflow.org/tutorials/audio/transfer_learning_audio)
- [Aumentación de datos de audio](https://www.tensorflow.org/io/tutorials/audio)
