# Detección de objetos con Android

Este tutorial le muestra cómo construir una app Android usando TensorFlow Lite para detectar continuamente objetos en fotogramas capturados por la cámara de un dispositivo. Esta aplicación está diseñada para un dispositivo Android físico. Si está actualizando un proyecto existente, puede usar el ejemplo de código como referencia y pasar a las instrucciones para [modificar su proyecto](#add_dependencies).

![Demo animada de detección de objetos](https://storage.googleapis.com/download.tensorflow.org/tflite/examples/obj_detection_cat.gif){: .attempt-right width="250px"}

## Visión general de la detección de objetos

La *detección de objetos* es la tarea de aprendizaje automático que consiste en identificar la presencia y la ubicación de varias clases de objetos dentro de una imagen. Un modelo de detección de objetos se entrena en un conjunto de datos que contiene un conjunto de objetos conocidos.

El modelo entrenado recibe fotogramas de imágenes como entrada e intenta clasificar los objetos de las imágenes a partir del conjunto de clases conocidas para cuyo reconocimiento fue entrenado. Para cada fotograma de imagen, el modelo de detección de objetos emite una lista de los objetos que detecta, la ubicación de un cuadro delimitador para cada objeto y una puntuaciones que indican la confianza en que el objeto se clasifique correctamente.

## Modelos y conjunto de datos

Este tutorial usa modelos que fueron entrenados usando el conjunto de datos [COCO](http://cocodataset.org/). COCO es un conjunto de datos de detección de objetos a gran escala que contiene 330,000 imágenes, 1.5 millones de instancias de objetos y 80 categorías de objetos.

Tiene la opción de usar uno de los siguientes modelos preentrenados:

- [EfficientDet-Lite0](https://tfhub.dev/tensorflow/lite-model/efficientdet/lite0/detection/metadata/1) *[Recomendado]*: un modelo ligero de detección de objetos con un extractor de características BiFPN, un predictor de caja compartida y una pérdida focal. La mAP (precisión media) para el conjunto de datos de validación COCO 2017 es del 25.69 %.

- [EfficientDet-Lite1](https://tfhub.dev/tensorflow/lite-model/efficientdet/lite1/detection/metadata/1): un modelo de detección de objetos EfficientDet de tamaño medio. El mAP para el conjunto de datos de validación COCO 2017 es del 30.55 %.

- [EfficientDet-Lite2](https://tfhub.dev/tensorflow/lite-model/efficientdet/lite2/detection/metadata/1): un modelo de detección de objetos EfficientDet más amplio. El mAP para el conjunto de datos de validación COCO 2017 es del 33.97 %.

- [MobileNetV1-SSD](https://tfhub.dev/tensorflow/lite-model/ssd_mobilenet_v1/1/metadata/2): un modelo extremadamente ligero optimizado para trabajar con TensorFlow Lite para la detección de objetos. El mAP para el conjunto de datos de validación COCO 2017 es del 21 %.

Para este tutorial, el modelo *EfficientDet-Lite0* consigue un buen equilibrio entre tamaño y precisión.

La descarga, extracción y colocación de los modelos en la carpeta de activos se administra automáticamente mediante el archivo `download.gradle`, que se ejecuta en el momento de la compilación. No es necesario descargar manualmente los modelos TFLite en el proyecto.

## Configurar y ejecutar el ejemplo

Para configurar la app de detección de objetos, descargue el ejemplo de [GitHub](https://github.com/tensorflow/examples/tree/master/lite/examples/object_detection/android) y ejecútelo usando [Android Studio](https://developer.android.com/studio/). Las siguientes secciones de este tutorial exploran las secciones relevantes del ejemplo de código, para que pueda aplicarlas a sus propias apps Android.

### Requisitos del sistema

- <a>Android Studio</a> versión 2021.1.1 (Bumblebee) o superior.
- Android SDK versión 31 o superior
- Dispositivo Android con una versión mínima del sistema operativo de SDK 24 (Android 7.0 - Nougat) con el modo de desarrollador activado.

Nota: Este ejemplo usa una cámara, así que ejecútelo en un dispositivo Android físico.

### Obtenga el código del ejemplo

Cree una copia local del código de ejemplo. Usará este código para crear un proyecto en Android Studio y ejecutar la aplicación de ejemplo.

Para clonar y configurar el código de ejemplo:

1. Clone the git repository
    <pre class="devsite-click-to-copy">
        git clone https://github.com/tensorflow/examples.git
        </pre>
2. Opcionalmente, configure su instancia git para usar sparse checkout, de modo que sólo tenga los archivos para la app de ejemplo de detección de objetos:
    <pre class="devsite-click-to-copy">
        cd examples
        git sparse-checkout init --cone
        git sparse-checkout set lite/examples/object_detection/android
        </pre>

### Importe y ejecute el proyecto

Cree un proyecto a partir del código de ejemplo descargado, compile el proyecto y ejecútelo.

Para importar y generar el proyecto de código de ejemplo:

1. Inicie [Android Studio](https://developer.android.com/studio).
2. En Android Studio, seleccione **Archivo &gt; Nuevo &gt; Importar proyecto**.
3. Navegue hasta el directorio de código de ejemplo que contiene el archivo build.gradle (`.../examples/lite/examples/object_detection/android/build.gradle`) y seleccione ese directorio.
4. Si Android Studio solicita una Sincronización Gradle, seleccione OK.
5. Asegúrese de que su dispositivo Android está conectado a la computadora y de que el modo de desarrollador está activado. Haga clic en la flecha verde `Run`.

Si selecciona el directorio correcto, Android Studio crea un nuevo proyecto y lo genera. Este proceso puede tardar unos minutos, dependiendo de la velocidad de su computadora y de si ha usado Android Studio para otros proyectos. Cuando la compilación se completa, Android Studio muestra un mensaje `GENERACIÓN EXITOSA` en el panel de estado **Resultado de generación**.

Nota: El código de ejemplo está generado con Android Studio 4.2.2, pero funciona con versiones anteriores de Studio. Si está usando una versión anterior de Android Studio puede intentar ajustar el número de versión del plugin de Android para que se complete la generación, en lugar de actualizar Studio.

**Opcional:** Para corregir errores de compilación actualizando la versión del plugin de Android:

1. Abra el archivo build.gradle en el directorio del proyecto.

2. Cambie la versión de las herramientas de Android como se indica a continuación:

    ```
    // from: classpath
    'com.android.tools.build:gradle:4.2.2'
    // to: classpath
    'com.android.tools.build:gradle:4.1.2'
    ```

3. Sincronice el proyecto seleccionando **Archivo &gt; Sincronizar proyecto con archivos Gradle**.

Para ejecutar el proyecto:

1. Desde Android Studio, ejecute el proyecto seleccionando **Ejecutar &gt; Ejecutar...**.
2. Seleccione un dispositivo Android conectado con cámara para probar la app.

Las siguientes secciones le muestran las modificaciones que debe realizar en su proyecto existente para añadir esta funcionalidad a su propia app, usando esta app de ejemplo como punto de referencia.

## Añadir dependencias del proyecto {:#add_dependencies}

En su propia aplicación, debe añadir dependencias específicas del proyecto para ejecutar los modelos de aprendizaje automático de TensorFlow Lite, y acceder a las funciones de utilidad que convierten datos como imágenes, en un formato de datos de tensor que puede ser procesado por el modelo que está usando.

La app de ejemplo usa la [Librería de tareas para visión](../../inference_with_metadata/task_library/overview#supported_tasks) de TensorFlow Lite para permitir la ejecución del modelo de aprendizaje automático de detección de objetos. Las siguientes instrucciones explican cómo añadir las dependencias de librería necesarias a su propio proyecto de app para Android.

Las siguientes instrucciones explican cómo añadir las dependencias necesarias del proyecto a su propio proyecto de app para Android.

Para añadir dependencias de módulos:

1. En el módulo que usa TensorFlow Lite, actualice el archivo `build.gradle` del módulo para incluir las siguientes dependencias. En el código de ejemplo, este archivo se encuentra aquí: `...examples/lite/examples/object_detection/android/app/build.gradle` ([referencia de código](https://github.com/tensorflow/examples/blob/master/lite/examples/object_detection/android/app/build.gradle))

    ```
    dependencies {
      ...
      implementation 'org.tensorflow:tensorflow-lite-task-vision:0.4.0'
      // Import the GPU delegate plugin Library for GPU inference
      implementation 'org.tensorflow:tensorflow-lite-gpu-delegate-plugin:0.4.0'
      implementation 'org.tensorflow:tensorflow-lite-gpu:2.9.0'
    }
    ```

    El proyecto debe incluir la librería de tareas de visión (`tensorflow-lite-task-vision`). La librería de la unidad de procesamiento gráfico (GPU) (`tensorflow-lite-gpu-delegate-plugin`) aporta la infraestructura para ejecutar la app en la GPU, y Delegate (`tensorflow-lite-gpu`) aporta la lista de compatibilidad.

2. En Android Studio, sincronice las dependencias del proyecto seleccionando: **Archivo &gt; Sincronizar proyecto con archivos Gradle**.

## Inicializar el modelo ML

En su app para Android, debe inicializar el modelo de aprendizaje automático TensorFlow Lite con parámetros antes de ejecutar predicciones con el modelo. Estos parámetros de inicialización son consistentes en todos los modelos de detección de objetos y pueden incluir ajustes como umbrales mínimos de precisión para las predicciones.

Un modelo TensorFlow Lite incluye un archivo `.tflite` que contiene el código del modelo y, con frecuencia, incluye un archivo de etiquetas que contiene los nombres de las clases predichas por el modelo. En el caso de la detección de objetos, las clases son objetos como una persona, un perro, un gato o un coche.

Este ejemplo descarga varios modelos que se especifican en `download_models.gradle`, y la clase `ObjectDetectorHelper` nos ofrece un selector para los modelos:

```
val modelName =
  when (currentModel) {
    MODEL_MOBILENETV1 -> "mobilenetv1.tflite"
    MODEL_EFFICIENTDETV0 -> "efficientdet-lite0.tflite"
    MODEL_EFFICIENTDETV1 -> "efficientdet-lite1.tflite"
    MODEL_EFFICIENTDETV2 -> "efficientdet-lite2.tflite"
    else -> "mobilenetv1.tflite"
  }
```

Aspecto clave: Los modelos deben almacenarse en el directorio `src/main/assets` de su proyecto de desarrollo. La librería de tareas de TensorFlow Lite comprueba automáticamente este directorio cuando se especifica un nombre de archivo de modelo.

Para inicializar el modelo en su app:

1. Añada un archivo modelo `.tflite` al directorio `src/main/assets` de su proyecto de desarrollo, como por ejemplo: [EfficientDet-Lite0](https://tfhub.dev/tensorflow/lite-model/efficientdet/lite0/detection/metadata/1).

2. Ajuste una variable estática para el nombre del archivo de su modelo. En la app de ejemplo, se configura la variable `modelName` como `MODEL_EFFICIENTDETV0` para usar el modelo de detección EfficientDet-Lite0.

3. Configure las opciones del modelo, como el umbral de predicción, el tamaño del conjunto de resultados y, opcionalmente, los delegados de aceleración por hardware:

    ```
    val optionsBuilder =
      ObjectDetector.ObjectDetectorOptions.builder()
        .setScoreThreshold(threshold)
        .setMaxResults(maxResults)
    ```

4. Use la configuración de este objeto para construir un objeto [`ObjectDetector`](https://www.tensorflow.org/lite/api_docs/java/org/tensorflow/lite/task/vision/detector/ObjectDetector#createFromFile(Context,%20java.lang.String)) de TensorFlow Lite que contenga el modelo:

    ```
    objectDetector =
      ObjectDetector.createFromFileAndOptions(
        context, modelName, optionsBuilder.build())
    ```

El `setupObjectDetector` establece los siguientes parámetros del modelo:

- Umbral de detección
- Número máximo de resultados de detección
- Número de hilos de procesamiento que se usarán (`BaseOptions.builder().setNumThreads(numThreads)`)
- Modelo real (`modelName`)
- Objeto ObjectDetector (`objectDetector`)

### Configure el acelerador de hardware

Al inicializar un modelo TensorFlow Lite en su aplicación, puede usar las funciones de aceleración por hardware para acelerar los cálculos de predicción del modelo.

Los *delegados* de TensorFlow Lite son módulos de software que aceleran la ejecución de modelos de aprendizaje automático usando hardware de procesamiento especializado en un dispositivo móvil, como unidades de procesamiento gráfico (GPU), unidades de procesamiento de tensor (TPU) y procesadores de señal digital (DSP). Usar delegados para ejecutar modelos TensorFlow Lite es recomendable, pero no obligatorio.

El detector de objetos se inicializa mediante los ajustes actuales del hilo que lo está usando. Puede usar delegados de CPU y [NNAPI](../../android/delegates/nnapi) con detectores que se crean en el hilo principal y se usan en un hilo de fondo, pero el hilo que inicializó el detector debe usar el delegado de GPU.

Los delegados se fijan dentro de la función `ObjectDetectionHelper.setupObjectDetector()`:

```
when (currentDelegate) {
    DELEGATE_CPU -> {
        // Default
    }
    DELEGATE_GPU -> {
        if (CompatibilityList().isDelegateSupportedOnThisDevice) {
            baseOptionsBuilder.useGpu()
        } else {
            objectDetectorListener?.onError("GPU is not supported on this device")
        }
    }
    DELEGATE_NNAPI -> {
        baseOptionsBuilder.useNnapi()
    }
}
```

Para más información sobre cómo usar delegados de aceleración por hardware con TensorFlow Lite, consulte [Delegados de TensorFlow Lite](../../performance/delegates).

## Preparar los datos para el modelo

En su app para Android, su código ofrece datos al modelo para que los interprete transformando los datos existentes, como fotogramas de imágenes, en un formato de datos Tensor que pueda ser procesado por su modelo. Los datos en un Tensor que pase a un modelo deben tener unas dimensiones específicas, o forma, que coincida con el formato de datos usado para entrenar al modelo.

El modelo [EfficientDet-Lite0](https://tfhub.dev/tensorflow/lite-model/efficientdet/lite0/detection/metadata/1) usado en este ejemplo de código acepta Tensores que representan imágenes con una dimensión de 320 x 320, con tres canales (rojo, azul y verde) por pixel. Cada valor del tensor es un único byte entre 0 y 255. Por lo tanto, para ejecutar predicciones sobre nuevas imágenes, su app debe transformar esos datos de imagen en objetos de datos Tensor de ese tamaño y forma. La API de visión de la librería de tareas de TensorFlow Lite se encarga de la transformación de los datos por usted.

La app usa un objeto [`ImageAnalysis`](https://developer.android.com/training/camerax/analyze) para extraer imágenes de la cámara. Este objeto llama a la función `detectObject` con el mapa de bits de la cámara. Los datos son redimensionados y girados automáticamente por el `ImageProcessor` para que cumplan los requisitos de datos de imagen del modelo. A continuación, la imagen se convierte en un objeto [`TensorImage`](https://www.tensorflow.org/lite/api_docs/java/org/tensorflow/lite/support/image/TensorImage).

Para preparar los datos del subsistema de la cámara para ser procesados por el modelo ML:

1. Genere un objeto `ImageAnalysis` para extraer las imágenes en el formato requerido:

    ```
    imageAnalyzer =
        ImageAnalysis.Builder()
            .setTargetAspectRatio(AspectRatio.RATIO_4_3)
            .setTargetRotation(fragmentCameraBinding.viewFinder.display.rotation)
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .setOutputImageFormat(OUTPUT_IMAGE_FORMAT_RGBA_8888)
            .build()
            ...
    ```

2. Conecte el analizador al subsistema de la cámara y cree un búfer de mapa de bits para contener los datos recibidos de la cámara:

    ```
    .also {
      it.setAnalyzer(cameraExecutor) {
        image -> if (!::bitmapBuffer.isInitialized)
        { bitmapBuffer = Bitmap.createBitmap( image.width, image.height,
        Bitmap.Config.ARGB_8888 ) } detectObjects(image)
        }
      }
    ```

3. Extraiga los datos específicos de la imagen que necesita el modelo y pase la información de rotación de la imagen:

    ```
    private fun detectObjects(image: ImageProxy) {
      //Copy out RGB bits to the shared bitmap buffer
      image.use {bitmapBuffer.copyPixelsFromBuffer(image.planes[0].buffer) }
        val imageRotation = image.imageInfo.rotationDegrees
        objectDetectorHelper.detect(bitmapBuffer, imageRotation)
      }
    ```

4. Complete toda transformación final de los datos y añada los datos de la imagen a un objeto `TensorImage`, como se muestra en el método `ObjectDetectorHelper.detect()` de la app de ejemplo:

    ```
    val imageProcessor = ImageProcessor.Builder().add(Rot90Op(-imageRotation / 90)).build()
    // Preprocess the image and convert it into a TensorImage for detection.
    val tensorImage = imageProcessor.process(TensorImage.fromBitmap(image))
    ```

Nota: Cuando extraiga información de la imagen del subsistema de cámara de Android, asegúrese de obtener una imagen en formato RGB. Este formato es requerido por la clase ImageProcessor de TensorFlow Lite que se usa para preparar la imagen para su análisis por un modelo. Si la imagen en formato RGB contiene un canal Alfa, se ignorarán esos datos de transparencia.

## Ejecute predicciones

En su app Android, una vez creado un objeto TensorImage con datos de imagen en el formato correcto, puede ejecutar el modelo contra esos datos para producir una predicción, o *inferencia*.

En la clase `fragments/CameraFragment.kt` de la app de ejemplo, el objeto `imageAnalyzer` dentro de la función `bindCameraUseCases` pasa automáticamente los datos al modelo para las predicciones cuando la app se conecta a la cámara.

La app usa el método `cameraProvider.bindToLifecycle()` para manejar el selector de cámara, la ventana de vista previa y el procesamiento del modelo ML. La clase `ObjectDetectorHelper.kt` se encarga de pasar los datos de la imagen al modelo. Para ejecutar el modelo y generar predicciones a partir de los datos de la imagen:

- Ejecute la predicción pasando los datos de la imagen a su función de predicción:

    ```
    val results = objectDetector?.detect(tensorImage)
    ```

El objeto Intérprete de TensorFlow Lite recibe estos datos, los ejecuta contra el modelo y produce una lista de predicciones. Para un procesamiento continuo de los datos por el modelo, use el método `runForMultipleInputsOutputs()` para que los objetos Intérprete no sean creados y luego eliminados por el sistema para cada ejecución de predicción.

## Manejar la salida del modelo

En su app Android, después de cotejar los datos de la imagen con el modelo de detección de objetos, éste produce una lista de predicciones que el código de su app debe manejar ejecutando lógica de negocio adicional, mostrando resultados al usuario o realizando otras acciones.

El resultado de cualquier modelo TensorFlow Lite varía en función del número de predicciones que produce (una o muchas) y de la información descriptiva de cada predicción. En el caso de un modelo de detección de objetos, las predicciones suelen incluir datos para un cuadro delimitador que indica dónde se detecta un objeto en la imagen. En el código de ejemplo, los resultados se pasan a la función `onResults` en `CameraFragment.kt`, que actúa como DetectorListener en el proceso de detección de objetos.

```
interface DetectorListener {
  fun onError(error: String)
  fun onResults(
    results: MutableList<Detection>?,
    inferenceTime: Long,
    imageHeight: Int,
    imageWidth: Int
  )
}
```

Para el modelo usado en este ejemplo, cada predicción incluye una ubicación de la caja delimitadora para el objeto, una etiqueta para el objeto y una puntuación de predicción entre 0 y 1 como Float que representa la confianza de la predicción, siendo 1 el índice de confianza más alto. En general, las predicciones con una puntuación inferior al 50 % (0.5) se consideran no concluyentes. No obstante, la forma en que gestione los resultados de predicción con una puntuación baja depende de usted y de las necesidades de su aplicación.

Para manejar los resultados de la predicción del modelo:

1. Use un patrón de receptor de eventos para pasar los resultados al código de su app o a los objetos de la interfaz de usuario. La app de ejemplo usa este patrón para pasar resultados de detección del objeto `ObjectDetectorHelper` al objeto `CameraFragment`:

    ```
    objectDetectorListener.onResults(
    // instance of CameraFragment
        results,
        inferenceTime,
        tensorImage.height,
        tensorImage.width)
    ```

2. Actúe sobre los resultados, por ejemplo mostrando la predicción al usuario. El ejemplo dibuja un recuadro sobre el objeto <code>CameraPreview</code> para mostrar el resultado:

    ```
    override fun onResults(
      results: MutableList<Detection>?,
      inferenceTime: Long,
      imageHeight: Int,
      imageWidth: Int
    ) {
        activity?.runOnUiThread {
            fragmentCameraBinding.bottomSheetLayout.inferenceTimeVal.text =
                String.format("%d ms", inferenceTime)

            // Pass necessary information to OverlayView for drawing on the canvas
            fragmentCameraBinding.overlay.setResults(
                results ?: LinkedList<Detection>(),
                imageHeight,
                imageWidth
            )

            // Force a redraw
            fragmentCameraBinding.overlay.invalidate()
        }
    }
    ```

Una vez que el modelo ha devuelto un resultado de predicción, su aplicación puede actuar sobre esa predicción presentando el resultado a su usuario o ejecutando lógica adicional. En el caso del código de ejemplo, la aplicación dibuja un cuadro delimitador alrededor del objeto identificado y muestra el nombre de la clase en pantalla.

## Siguientes pasos

- Explore varios usos de TensorFlow Lite en los [ejemplos](../../examples).
- Aprenda más sobre cómo usar modelos de aprendizaje automático con TensorFlow Lite en la sección [Modelos](../../models).
- Aprenda más sobre cómo implementar el aprendizaje automático en su Aplicación Móvil en la [Guía para desarrolladores de TensorFlow Lite](../../guide).
