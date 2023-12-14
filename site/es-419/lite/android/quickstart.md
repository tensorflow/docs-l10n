# Inicio rápido para Android

Esta página le muestra cómo crear una app Android con TensorFlow Lite para analizar una imagen de cámara en directo e identificar objetos. Este caso de uso de aprendizaje automático se denomina *detección de objetos*. La app de ejemplo usa la [Librería de tareas para visión](../inference_with_metadata/task_library/overview#supported_tasks) de TensorFlow Lite a través de [los servicios de Google Play](./play_services) para permitir la ejecución del modelo de aprendizaje automático de detección de objetos, que es el enfoque recomendado para construir una aplicación ML con TensorFlow Lite.

<aside class="note"><b>Condiciones:</b> Al acceder o usar las API de TensorFlow Lite en los servicios de Google Play, usted acepta los <a href="./play_services#tos">Términos del servicio</a>. Lea y comprenda todos los términos y políticas aplicables antes de acceder a las API.</aside>

![Demo animada de detección de objetos](https://storage.googleapis.com/download.tensorflow.org/tflite/examples/obj_detection_cat.gif){:.attempt-right width="250px"}

## Configurar y ejecutar el ejemplo

Para la primera parte de este ejercicio, descargue el [código de ejemplo](https://github.com/tensorflow/examples/tree/master/lite/examples/object_detection/android_play_services) de GitHub y ejecútelo usando [Android Studio](https://developer.android.com/studio/). Las siguientes secciones de este documento exploran las secciones relevantes del código de ejemplo, para que pueda aplicarlas a sus propias apps Android. Necesita tener instaladas las siguientes versiones de estas herramientas:

- Android Studio 4.2 o superior
- Android SDK versión 21 o superior

Nota: Este ejemplo usa la cámara, por lo que deberá ejecutarlo en un dispositivo Android físico.

### Obtenga el código del ejemplo

Cree una copia local del código de ejemplo para poder compilarlo y ejecutarlo.

Para clonar y configurar el código de ejemplo:

1. Clone el repositorio git
    <pre class="devsite-click-to-copy">    git clone https://github.com/tensorflow/examples.git
        </pre>
2. Configure su instancia git para usar sparse checkout, de modo que sólo tenga los archivos para la app de ejemplo de detección de objetos:
    <pre class="devsite-click-to-copy">    cd examples
        git sparse-checkout init --cone
        git sparse-checkout set lite/examples/object_detection/android_play_services
        </pre>

### Importe y ejecute el proyecto

Use Android Studio para crear un proyecto a partir del código de ejemplo descargado, compile el proyecto y ejecútelo.

Para importar y generar el proyecto de código de ejemplo:

1. Inicie [Android Studio](https://developer.android.com/studio).
2. En la página **Bienvenida** de Android Studio, seleccione **Importar proyecto**, o seleccione **Archivo &gt; Nuevo &gt; Importar proyecto**.
3. Navegue hasta el directorio de código de ejemplo que contiene el archivo build.gradle (`...examples/lite/examples/object_detection/android_play_services/build.gradle`) y seleccione ese directorio.

Después de seleccionar este directorio, Android Studio crea un nuevo proyecto y lo construye. Cuando la compilación se llena, Android Studio muestra un mensaje `BUILD SUCCESSFUL` en el panel de estado **Build Output**.

Para ejecutar el proyecto:

1. Desde Android Studio, ejecute el proyecto seleccionando **Ejecutar &gt; Ejecutar...** y **MainActivity**.
2. Seleccione un dispositivo Android conectado con cámara para probar la app.

## Cómo funciona la app de ejemplo

La app de ejemplo usa un modelo de detección de objetos preentrenado, como [mobilenetv1.tflite](https://tfhub.dev/tensorflow/lite-model/ssd_mobilenet_v1/1/metadata/2?lite-format=tflite), en formato TensorFlow Lite busca objetos en un flujo de vídeo en directo desde la cámara de un dispositivo Android. El código para esta función se encuentra principalmente en estos archivos:

- [ObjectDetectorHelper.kt](https://github.com/tensorflow/examples/blob/master/lite/examples/object_detection/android_play_services/app/src/main/java/org/tensorflow/lite/examples/objectdetection/ObjectDetectorHelper.kt): Inicializa el entorno runtime, habilita la aceleración por hardware y ejecuta el modelo ML de detección de objetos.
- [CameraFragment.kt](https://github.com/tensorflow/examples/blob/master/lite/examples/object_detection/android_play_services/app/src/main/java/org/tensorflow/lite/examples/objectdetection/fragments/CameraFragment.kt): Construye el flujo de datos de la imagen de la cámara, prepara los datos para el modelo y muestra los resultados de la detección de objetos.

Nota: Esta app de ejemplo usa la [Librería de tareas](../inference_with_metadata/task_library/overview#supported_tasks) de TensorFlow Lite, que ofrece APIs fáciles de usar y específicas de la tarea para realizar operaciones comunes de aprendizaje automático. Para apps con necesidades más específicas y funciones de ML personalizadas, considere usar la [Interpreter API](https://www.tensorflow.org/lite/api_docs/java/org/tensorflow/lite/InterpreterApi).

Las siguientes secciones le muestran los componentes clave de estos archivos de código, para que pueda modificar una app de Android y añadirle esta funcionalidad.

## Genere la app {:#build_app}

Las siguientes secciones explican los pasos clave para construir su propia app Android y ejecutar el modelo mostrado en la app de ejemplo. Estas instrucciones usan la app de ejemplo mostrada anteriormente como punto de referencia.

Nota: Para seguir estas instrucciones y crear su propia app, cree un [proyecto Android básico](https://developer.android.com/studio/projects/create-project) utilizando Android Studio.

### Añadir dependencias del proyecto {:#add_dependencies}

En su app básica para Android, añada las dependencias del proyecto para ejecutar modelos de aprendizaje automático TensorFlow Lite y acceder a las funciones de utilidad de datos ML. Estas funciones de utilidades convierten datos como imágenes en un formato de datos de tensor que puede ser procesado por un modelo.

La app de ejemplo usa la [Librería de tareas para visión](../inference_with_metadata/task_library/overview#supported_tasks)[ TensorFlow Lite de ](../inference_with_metadata/task_library/overview#supported_tasks)[Servicios de Google Play](./play_services) para permitir la ejecución del modelo de aprendizaje automático de detección de objetos. Las siguientes instrucciones explican cómo añadir las dependencias de librería necesarias a su propio proyecto de app para Android.

Para añadir dependencias de módulos:

1. En el módulo de la app para Android que usa TensorFlow Lite, actualice el archivo `build.gradle` del módulo para incluir las siguientes dependencias. En el código de ejemplo, este archivo se encuentra aquí: `...examples/lite/examples/object_detection/android_play_services/app/build.gradle`
    ```
    ...
    dependencies {
    ...
        // Tensorflow Lite dependencies
        implementation 'org.tensorflow:tensorflow-lite-task-vision-play-services:0.4.2'
        implementation 'com.google.android.gms:play-services-tflite-gpu:16.1.0'
    ...
    }
    ```
2. En Android Studio, sincronice las dependencias del proyecto seleccionando: **Archivo &gt; Sincronizar proyecto con archivos Gradle**.

### Inicialice los servicios de Google Play

Cuando utilice [servicios de Google Play](./play_services) para ejecutar modelos TensorFlow Lite, deberá inicializar el servicio antes de poder usarlo. Si desea usar el soporte de aceleración por hardware con el servicio, como la aceleración por GPU, también debe habilitar dicho soporte como parte de esta inicialización.

Para inicializar TensorFlow Lite con los servicios de Google Play:

1. Cree un objeto `TfLiteInitializationOptions` y modifíquelo para habilitar la compatibilidad con la GPU:

    ```
    val options = TfLiteInitializationOptions.builder()
        .setEnableGpuDelegateSupport(true)
        .build()
    ```

2. Utilice el método `TfLiteVision.initialize()` para habilitar el uso del runtime de los servicios Play y configure un receptor de eventos para verificar que se ha cargado correctamente:

    ```
    TfLiteVision.initialize(context, options).addOnSuccessListener {
        objectDetectorListener.onInitialized()
    }.addOnFailureListener {
        // Called if the GPU Delegate is not supported on the device
        TfLiteVision.initialize(context).addOnSuccessListener {
            objectDetectorListener.onInitialized()
        }.addOnFailureListener{
            objectDetectorListener.onError("TfLiteVision failed to initialize: "
                    + it.message)
        }
    }
    ```

### Inicializar el intérprete del modelo ML

Inicialice el intérprete del modelo de aprendizaje automático de TensorFlow Lite cargando el archivo del modelo y configurando los parámetros del mismo. Un modelo TensorFlow Lite incluye un archivo `.tflite` que contiene el código del modelo. Debe almacenar sus modelos en el directorio `src/main/assets` de su proyecto de desarrollo, por ejemplo:

```
.../src/main/assets/mobilenetv1.tflite`
```

Consejo: El código del intérprete de la librería de tareas busca automáticamente los modelos en el directorio `src/main/assets` si no se especifica una ruta de archivo.

Para inicializar el modelo:

1. Añada un archivo modelo `.tflite` al directorio `src/main/assets` de su proyecto de desarrollo, como [ssd_mobilenet_v1](https://tfhub.dev/tensorflow/lite-model/ssd_mobilenet_v1/1/metadata/2).
2. Configure la variable `modelName` para especificar el nombre del archivo de su modelo ML:
    ```
    val modelName = "mobilenetv1.tflite"
    ```
3. Configure las opciones para el modelo, como el umbral de predicción y el tamaño del conjunto de resultados:
    ```
    val optionsBuilder =
        ObjectDetector.ObjectDetectorOptions.builder()
            .setScoreThreshold(threshold)
            .setMaxResults(maxResults)
    ```
4. Habilite la aceleración de GPU con las opciones y permita que el código maneje sus fallos con elegancia si la aceleración no es compatible con el dispositivo:
    ```
    try {
        optionsBuilder.useGpu()
    } catch(e: Exception) {
        objectDetectorListener.onError("GPU is not supported on this device")
    }

    ```
5. Usar la configuración de este objeto para construir un objeto [`ObjectDetector`](https://www.tensorflow.org/lite/api_docs/java/org/tensorflow/lite/task/vision/detector/ObjectDetector#createFromFile(Context,%20java.lang.String)) de TensorFlow Lite que contenga el modelo:
    ```
    objectDetector =
        ObjectDetector.createFromFileAndOptions(
            context, modelName, optionsBuilder.build())
    ```

Para más información sobre cómo usar delegados de aceleración por hardware con TensorFlow Lite, consulte [Delegados de TensorFlow Lite](../performance/delegates).

### Preparar los datos para el modelo

Usted prepara los datos para su interpretación por el modelo al transformar los datos existentes, como las imágenes, en el formato de datos [Tensor](../api_docs/java/org/tensorflow/lite/Tensor), para que puedan ser procesados por su modelo. Los datos en un Tensor deben tener dimensiones específicas, es decir, una forma, que coincida con el formato de los datos usados para el entrenamiento del modelo. Según el modelo que use, puede que tenga que transformar los datos para que se ajusten a lo que espera el modelo. La app de ejemplo usa un objeto [`ImageAnalysis`](https://developer.android.com/reference/androidx/camera/core/ImageAnalysis) para extraer cuadros de imagen del subsistema de la cámara.

Para preparar los datos para su procesamiento por el modelo:

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
            it.setAnalyzer(cameraExecutor) { image ->
                if (!::bitmapBuffer.isInitialized) {
                    bitmapBuffer = Bitmap.createBitmap(
                        image.width,
                        image.height,
                        Bitmap.Config.ARGB_8888
                    )
                }
                detectObjects(image)
            }
        }
    ```
3. Extraiga los datos específicos de la imagen que necesita el modelo y pase la información de rotación de la imagen:
    ```
    private fun detectObjects(image: ImageProxy) {
        // Copy out RGB bits to the shared bitmap buffer
        image.use { bitmapBuffer.copyPixelsFromBuffer(image.planes[0].buffer) }
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

### Ejecute predicciones

Una vez creado un objeto [TensorImage](../api_docs/java/org/tensorflow/lite/support/image/TensorImage) con datos de imagen en el formato correcto, puede ejecutar el modelo contra esos datos para producir una predicción, o *inferencia*. En la app de ejemplo, este código está contenido en el método `ObjectDetectorHelper.detect()`.

Para ejecutar un modelo y generar predicciones a partir de datos de imágenes:

- Ejecute la predicción pasando los datos de la imagen a su función de predicción:
    ```
    val results = objectDetector?.detect(tensorImage)
    ```

### Manejar la salida del modelo

Después de contrastar los datos de la imagen con el modelo de detección de objetos, éste produce una lista de resultados de predicción que el código de su app debe manejar ejecutando lógica de negocio adicional, mostrando los resultados al usuario o realizando otras acciones. El modelo de detección de objetos de la app de ejemplo produce una lista de predicciones y cuadros delimitadores para los objetos detectados. En la app de ejemplo, los resultados de la predicción se pasan a un objeto receptor para su posterior procesamiento y visualización al usuario.

Para manejar los resultados de la predicción del modelo:

1. Use un patrón de receptor de eventos para pasar los resultados al código de su app o a los objetos de la interfaz de usuario. La app de ejemplo usa este patrón para pasar resultados de detección del objeto `ObjectDetectorHelper` al objeto `CameraFragment`:
    ```
    objectDetectorListener.onResults( // instance of CameraFragment
        results,
        inferenceTime,
        tensorImage.height,
        tensorImage.width)
    ```
2. Actúe sobre los resultados, por ejemplo mostrando la predicción al usuario. La app de ejemplo dibuja un recuadro sobre el objeto `CameraPreview` para mostrar el resultado:
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

## Siguientes pasos

- Aprenda más sobre las [ APIs de la librería de tareas](../inference_with_metadata/task_library/overview#supported_tasks)
- Aprenda más sobre las [APIs del Intérprete](https://www.tensorflow.org/lite/api_docs/java/org/tensorflow/lite/InterpreterApi).
- Explore los usos de TensorFlow Lite en los [ejemplos](../examples).
- Aprenda más sobre cómo usar y construir modelos de aprendizaje automático con TensorFlow Lite en la sección [Modelos](../models).
- Aprenda más sobre cómo implementar el aprendizaje automático en su Aplicación Móvil en la [Guía para desarrolladores de TensorFlow Lite](../guide).
