# Reconocimiento de sonidos y palabras para Android

Este tutorial le muestra cómo usar TensorFlow Lite con modelos de aprendizaje automático preconstruidos para reconocer sonidos y palabras habladas en una app Android. Los modelos de clasificación de audio como los que se muestran en este tutorial pueden usarse para detectar actividad, identificar acciones o reconocer comandos de voz.

![Demo animada de reconocimiento de audio](https://storage.googleapis.com/download.tensorflow.org/tflite/examples/audio_classification.gif){: .attempt-right} Este tutorial le muestra cómo descargar el código de ejemplo, cargar el proyecto en [Android Studio](https://developer.android.com/studio/), y le explica las partes clave del código de ejemplo para que pueda empezar a añadir esta funcionalidad a su propia app. El código de la app de ejemplo usa la [Librería de tareas TensorFlow para audio](https://www.tensorflow.org/lite/inference_with_metadata/task_library/audio_classifier), que se encarga de la mayor parte de la grabación y el preprocesamiento de los datos de audio. Para obtener más información sobre cómo se preprocesa el audio para usarlo con modelos de aprendizaje automático, consulte [Preparación y aumentación de datos de audio](https://www.tensorflow.org/io/tutorials/audio).

## Clasificación de audio con aprendizaje automático

El modelo de aprendizaje automático de este tutorial reconoce sonidos o palabras a partir de muestras de audio grabadas con un micrófono en un dispositivo Android. La app de ejemplo de este tutorial le permite alternar entre el [YAMNet/clasificador](https://tfhub.dev/google/lite-model/yamnet/classification/tflite/1), un modelo que reconoce sonidos, y un modelo que reconoce palabras habladas específicas, que fue [entrenado](https://www.tensorflow.org/lite/models/modify/model_maker/speech_recognition) usando la herramienta [Model Maker](https://www.tensorflow.org/lite/models/modify/model_maker) de TensorFlow Lite. Los modelos ejecutan predicciones sobre clips de audio que contienen 15600 muestras individuales por clip y tienen una duración aproximada de 1 segundo.

## Configurar y ejecutar el ejemplo

Para la primera parte de este tutorial, descargue el ejemplo de GitHub y ejecútelo usando Android Studio. Las siguientes secciones de este tutorial exploran las secciones relevantes del ejemplo, para que pueda aplicarlas a sus propias apps Android.

### Requisitos del sistema

- [Android Studio](https://developer.android.com/studio/index.html) versión 2021.1.1 (Bumblebee) o superior.
- Android SDK versión 31 o superior
- Dispositivo Android con una versión mínima del sistema operativo de SDK 24 (Android 7.0 - Nougat) con el modo de desarrollador activado.

### Obtenga el código del ejemplo

Cree una copia local del código de ejemplo. Usará este código para crear un proyecto en Android Studio y ejecutar la aplicación de ejemplo.

Para clonar y configurar el código de ejemplo:

1. Clone el repositorio git
    <pre class="devsite-click-to-copy">
        git clone https://github.com/tensorflow/examples.git
        </pre>
2. Opcionalmente, configure su instancia de git para usar sparse checkout, de forma que sólo tenga los archivos de la app de ejemplo:
    ```
    cd examples
    git sparse-checkout init --cone
    git sparse-checkout set lite/examples/audio_classification/android
    ```

### Importe y ejecute el proyecto

Cree un proyecto a partir del código de ejemplo descargado, compile el proyecto y ejecútelo.

Para importar y generar el proyecto de código de ejemplo:

1. Inicie [Android Studio](https://developer.android.com/studio).
2. En Android Studio, seleccione **Archivo &gt; Nuevo &gt; Importar proyecto**.
3. Navegue hasta el directorio de código de ejemplo que contiene el archivo `build.gradle` (`.../examples/lite/examples/audio_classification/android/build.gradle`) y seleccione ese directorio.

Si selecciona el directorio correcto, Android Studio crea un nuevo proyecto y lo construye. Este proceso puede tardar unos minutos, dependiendo de la velocidad de su computadora y de si ha usado Android Studio para otros proyectos. Cuando la compilación se llena, Android Studio muestra un mensaje `BUILD SUCCESSFUL` en el panel de estado **Build Output**.

Para ejecutar el proyecto:

1. Desde Android Studio, ejecute el proyecto seleccionando **Ejecutar &gt; Ejecutar 'app'**.
2. Seleccione un dispositivo Android conectado con micrófono para analizar la app.

Nota: Si usa un emulador para ejecutar la app, asegúrese de [habilitar la entrada de audio](https://developer.android.com/studio/releases/emulator#29.0.6-host-audio) desde la máquina host.

Las siguientes secciones le muestran las modificaciones que debe realizar en su proyecto existente para añadir esta funcionalidad a su propia app, usando esta app de ejemplo como punto de referencia.

## Añada las dependencias del proyecto

En su propia aplicación, debe añadir dependencias específicas del proyecto para ejecutar los modelos de aprendizaje automático de TensorFlow Lite y acceder a las funciones de utilidad que convierten los formatos de datos estándar, como el audio, en un formato de datos de tensor que pueda ser procesado por el modelo que esté usando.

La app de ejemplo usa las siguientes librerías TensorFlow Lite:

- [API de audio de la librería de tareas de TensorFlow Lite](https://www.tensorflow.org/lite/api_docs/java/org/tensorflow/lite/task/audio/classifier/package-summary): Ofrece las clases de entrada de datos de audio necesarias, la ejecución del modelo de aprendizaje automático y los resultados de salida del procesamiento del modelo.

Las siguientes instrucciones muestran cómo añadir las dependencias necesarias del proyecto a su propio proyecto de app para Android.

Para añadir dependencias de módulos:

1. En el módulo que usa TensorFlow Lite, actualice el archivo `build.gradle` del módulo para incluir las siguientes dependencias. En el código de ejemplo, este archivo se encuentra aquí: `.../examples/lite/examples/audio_classification/android/build.gradle`
    ```
    dependencies {
    ...
        implementation 'org.tensorflow:tensorflow-lite-task-audio'
    }
    ```
2. En Android Studio, sincronice las dependencias del proyecto seleccionando: **Archivo &gt; Sincronizar proyecto con archivos Gradle**.

## Inicializar el modelo ML

En su app para Android, debe inicializar el modelo de aprendizaje automático TensorFlow Lite con parámetros antes de ejecutar predicciones con el modelo. Estos parámetros de inicialización dependen del modelo y pueden incluir ajustes como umbrales mínimos predeterminados de precisión para las predicciones y etiquetas para las palabras o sonidos que el modelo puede reconocer.

Un modelo TensorFlow Lite incluye un archivo `*.tflite` que contiene el modelo. El archivo del modelo contiene la lógica de predicción y normalmente incluye [metadatos](../../models/convert/metadata) sobre cómo interpretar los resultados de la predicción, como los nombres de las clases de predicción. Los archivos de modelo deben almacenarse en el directorio `src/main/assets` de su proyecto de desarrollo, como en el ejemplo de código:

- `<project>/src/main/assets/yamnet.tflite`

Por comodidad y legibilidad del código, el ejemplo declara un objeto complementario que define los ajustes del modelo.

Para inicializar el modelo en su app:

1. Cree un objeto complementario para definir los ajustes del modelo:
    ```
    companion object {
      const val DISPLAY_THRESHOLD = 0.3f
      const val DEFAULT_NUM_OF_RESULTS = 2
      const val DEFAULT_OVERLAP_VALUE = 0.5f
      const val YAMNET_MODEL = "yamnet.tflite"
      const val SPEECH_COMMAND_MODEL = "speech.tflite"
    }
    ```
2. Cree los ajustes para el modelo generando un objeto `AudioClassifier.AudioClassifierOptions`:
    ```
    val options = AudioClassifier.AudioClassifierOptions.builder()
      .setScoreThreshold(classificationThreshold)
      .setMaxResults(numOfResults)
      .setBaseOptions(baseOptionsBuilder.build())
      .build()
    ```
3. Use este objeto de ajuste para construir un objeto TensorFlow Lite [`AudioClassifier`](https://www.tensorflow.org/lite/inference_with_metadata/task_library/audio_classifier) que contenga el modelo:
    ```
    classifier = AudioClassifier.createFromFileAndOptions(context, "yamnet.tflite", options)
    ```

### Habilitar la aceleración de hardware

Al inicializar un modelo TensorFlow Lite en su app, debe considerar la posibilidad de usar funciones de aceleración por hardware para acelerar los cálculos de predicción del modelo. Los [delegados](https://www.tensorflow.org/lite/performance/delegates) de TensorFlow Lite son módulos de software que aceleran la ejecución de modelos de aprendizaje automático usando hardware de procesamiento especializado en un dispositivo móvil, como las unidades de procesamiento gráfico (GPU) o las unidades de procesamiento tensorial (TPU). El código de ejemplo usa el delegado NNAPI para gestionar la aceleración por hardware de la ejecución del modelo:

```
val baseOptionsBuilder = BaseOptions.builder()
   .setNumThreads(numThreads)
...
when (currentDelegate) {
   DELEGATE_CPU -> {
       // Default
   }
   DELEGATE_NNAPI -> {
       baseOptionsBuilder.useNnapi()
   }
}
```

Usar delegados para ejecutar modelos TensorFlow Lite es recomendable, pero no obligatorio. Para saber más sobre cómo usar delegados con TensorFlow Lite, consulte [Delegados de TensorFlow Lite](https://www.tensorflow.org/lite/performance/delegates).

## Preparar los datos para el modelo

En su app para Android, su código ofrece datos al modelo para su interpretación transformando los datos existentes, como clips de audio, en un formato de datos [Tensor](https://www.tensorflow.org/lite/api_docs/java/org/tensorflow/lite/Tensor) que pueda ser procesado por su modelo. Los datos de un Tensor que usted pase a un modelo deben tener unas dimensiones específicas, o una forma, que coincida con el formato de los datos usados para entrenar al modelo.

El [modelo clasificador/YAMNet](https://tfhub.dev/google/lite-model/yamnet/classification/tflite/1) y los modelos personalizados de [comandos de voz](https://www.tensorflow.org/lite/models/modify/model_maker/speech_recognition) utilizados en este ejemplo de código aceptan objetos de datos Tensor que representan clips de audio de un solo canal, o mono, grabados a 16kHz en clips de 0.975 segundos (15600 muestreos). Al ejecutar predicciones sobre nuevos datos de audio, su app debe transformar esos datos de audio en objetos de datos Tensor de ese tamaño y forma. La [API de audio](https://www.tensorflow.org/lite/inference_with_metadata/task_library/audio_classifier) de la librería de tareas TensorFlow Lite se encarga de la transformación de los datos por usted.

En la clase de código de ejemplo `AudioClassificationHelper`, la app graba audio en directo desde los micrófonos del dispositivo usando un objeto [AudioRecord](https://developer.android.com/reference/android/media/AudioRecord) de Android. El código usa [AudioClassifier](https://www.tensorflow.org/lite/api_docs/java/org/tensorflow/lite/task/audio/classifier/AudioClassifier) para generar y configurar ese objeto con el fin de grabar audio a una frecuencia de muestreo apropiada para el modelo. El código también usa AudioClassifier para generar un objeto [TensorAudio](https://www.tensorflow.org/lite/api_docs/java/org/tensorflow/lite/support/audio/TensorAudio) para almacenar los datos de audio transformados. A continuación, el objeto TensorAudio se pasa al modelo para su análisis.

Para aportar datos de audio al modelo ML:

- Usa el objeto `AudioClassifier` para crear un objeto `TensorAudio` y un objeto `AudioRecord`:
    ```
    fun initClassifier() {
    ...
      try {
        classifier = AudioClassifier.createFromFileAndOptions(context, currentModel, options)
        // create audio input objects
        tensorAudio = classifier.createInputTensorAudio()
        recorder = classifier.createAudioRecord()
      }
    ```

Nota: Su app debe solicitar permiso para grabar audio usando el micrófono de un dispositivo Android. Consulte la clase `fragments/PermissionsFragment` del proyecto para ver un ejemplo. Para más información sobre la solicitud de permisos, consulte [Permisos en Android](https://developer.android.com/guide/topics/permissions/overview).

## Ejecute predicciones

En su app Android, una vez que haya conectado un objeto [AudioRecord](https://developer.android.com/reference/android/media/AudioRecord) y un objeto [TensorAudio](https://www.tensorflow.org/lite/api_docs/java/org/tensorflow/lite/support/audio/TensorAudio) a un objeto AudioClassifier, puede ejecutar el modelo contra esos datos para producir una predicción, o *inferencia*. El código de ejemplo de este tutorial ejecuta predicciones sobre clips de un flujo de entrada de audio grabado en directo a una velocidad específica.

La ejecución del modelo consume una cantidad significativa de recursos, por lo que es importante ejecutar las predicciones del modelo ML en un hilo separado y en segundo plano. La app de ejemplo usa un objeto `[ScheduledThreadPoolExecutor](https://developer.android.com/reference/java/util/concurrent/ScheduledThreadPoolExecutor)` para aislar el procesamiento del modelo de otras funciones de la app.

Los modelos de clasificación de audio que reconocen sonidos con un principio y un final claros, como las palabras, pueden producir predicciones más precisas en un flujo de audio entrante analizando clips de audio superpuestos. Este enfoque ayuda al modelo a evitar que se pierdan predicciones de palabras que se cortan al final de un clip. En la app de ejemplo, cada vez que ejecuta una predicción, el código toma el último clip de 0.975 segundos del búfer de grabación de audio y lo analiza. Puede hacer que el modelo analice clips de audio superpuestos ajustando el valor `interval` del grupo de ejecución del hilo de análisis del modelo a una longitud inferior a la de los clips analizados. Por ejemplo, si su modelo analiza clips de 1 segundo y usted establece el intervalo en 500 milisegundos, el modelo analizará la última mitad del clip anterior y 500 milisegundos de los nuevos datos de audio cada vez, creando una superposición de análisis de clips del 50 %.

Para empezar a ejecutar predicciones sobre los datos de audio:

1. Use el método `AudioClassificationHelper.startAudioClassification()` para iniciar la grabación de audio del modelo:
    ```
    fun startAudioClassification() {
      if (recorder.recordingState == AudioRecord.RECORDSTATE_RECORDING) {
        return
      }
      recorder.startRecording()
    }
    ```
2. Ajuste la frecuencia con la que el modelo genera una inferencia a partir de los clips de audio fijando un `interval` fijo en el objeto `ScheduledThreadPoolExecutor`:
    ```
    executor = ScheduledThreadPoolExecutor(1)
    executor.scheduleAtFixedRate(
      classifyRunnable,
      0,
      interval,
      TimeUnit.MILLISECONDS)
    ```
3. El objeto `classifyRunnable` del código anterior ejecuta el método `AudioClassificationHelper.classifyAudio()`, que carga los últimos datos de audio disponibles del grabador y realiza una predicción:
    ```
    private fun classifyAudio() {
      tensorAudio.load(recorder)
      val output = classifier.classify(tensorAudio)
      ...
    }
    ```

Precaución: No ejecute las predicciones del modelo ML en el hilo de ejecución principal de su aplicación. Hacerlo puede provocar que la interfaz de usuario de su app se vuelva lenta o no responda.

### Detenga el procesamiento de predicciones

Asegúrese de que el código de su app deja de realizar la clasificación de audio cuando el Fragmento o Actividad de procesamiento de audio de su app pierde el enfoque. La ejecución continua de un modelo de aprendizaje automático tiene un impacto significativo en la duración de la batería de un dispositivo Android. Use el método `onPause()` de la actividad o fragmento de Android asociado a la clasificación de audio para detener la grabación de audio y el procesamiento de predicciones.

Para detener la grabación y la clasificación de audio:

- Use el método `AudioClassificationHelper.stopAudioClassification()` para detener la grabación y la ejecución del modelo, como se muestra a continuación en la clase `AudioFragment`:
    ```
    override fun onPause() {
      super.onPause()
      if (::audioHelper.isInitialized ) {
        audioHelper.stopAudioClassification()
      }
    }
    ```

## Manejar la salida del modelo

En su app Android, después de procesar un clip de audio, el modelo produce una lista de predicciones que el código de su app debe manejar con la ejecución de lógica de negocio adicional, mostrando los resultados al usuario o realizando otras acciones.  La salida de cualquier modelo TensorFlow Lite dado varía en términos del número de predicciones que produce (una o muchas), y la información descriptiva de cada predicción. En el caso de los modelos de la app de ejemplo, las predicciones son una lista de sonidos reconocidos o de palabras. El objeto de opciones AudioClassifier usado en el ejemplo de código le permite fijar el número máximo de predicciones con el método `setMaxResults()`, como se muestra en la sección [Inicializar el modelo ML](#Initialize_the_ML_model).

Para obtener los resultados de la predicción del modelo:

1. Obtenga los resultados del método `classify()` del objeto [AudioClassifier](https://www.tensorflow.org/lite/api_docs/java/org/tensorflow/lite/task/audio/classifier/AudioClassifier) y páselos al objeto receptor de eventos (referencia de código):
    ```
    private fun classifyAudio() {
      ...
      val output = classifier.classify(tensorAudio)
      listener.onResult(output[0].categories, inferenceTime)
    }
    ```
2. Use la función onResult() del receptor de eventos para gestionar la salida ejecutando la lógica empresarial o mostrando los resultados al usuario:
    ```
    private val audioClassificationListener = object : AudioClassificationListener {
      override fun onResult(results: List<Category>, inferenceTime: Long) {
        requireActivity().runOnUiThread {
          adapter.categoryList = results
          adapter.notifyDataSetChanged()
          fragmentAudioBinding.bottomSheetLayout.inferenceTimeVal.text =
            String.format("%d ms", inferenceTime)
        }
      }
    ```

El modelo usado en este ejemplo genera una lista de predicciones con una etiqueta para el sonido o la palabra clasificada, y una puntuación de predicción entre 0 y 1 como Float que representa la confianza de la predicción, siendo 1 el índice de confianza más alto. En general, las predicciones con una puntuación inferior al 50 % (0.5) se consideran no concluyentes. No obstante, la forma en que gestione los resultados de predicción con una puntuación baja depende de usted y de las necesidades de su aplicación.

Una vez que el modelo ha devuelto un conjunto de resultados de predicción, su aplicación puede actuar sobre esas predicciones presentando el resultado a su usuario o ejecutando lógica adicional. En el caso del código de ejemplo, la aplicación enumera los sonidos o palabras identificados en la interfaz de usuario de la app.

## Siguientes pasos

Puede encontrar modelos adicionales de TensorFlow Lite para el procesamiento de audio en [TensorFlow Hub](https://tfhub.dev/s?deployment-format=lite&module-type=audio-embedding,audio-pitch-extraction,audio-event-classification,audio-stt) y a través de la página [Guía de modelos preentrenados](https://www.tensorflow.org/lite/models/trained). Para saber más sobre cómo implementar el aprendizaje automático en su Aplicación Móvil con TensorFlow Lite, consulte la [Guía para desarrolladores de TensorFlow Lite](https://www.tensorflow.org/lite/guide).
