# Clasificación de textos con Android

Este tutorial le muestra cómo construir una aplicación Android usando TensorFlow Lite para clasificar texto en lenguaje natural. Esta aplicación está diseñada para un dispositivo Android físico, pero también se puede ejecutar en un emulador de dispositivo.

La [aplicación de ejemplo](https://github.com/tensorflow/examples/tree/master/lite/examples/text_classification/android) usa TensorFlow Lite para clasificar texto como positivo o negativo, usando la [librería de tareas para lenguaje natural (NL)](../../inference_with_metadata/task_library/overview#supported_tasks) para permitir la ejecución de los modelos de aprendizaje automático de clasificación de texto.

Si está actualizando un proyecto existente, puede usar la aplicación de ejemplo como referencia o plantilla. Si desea instrucciones sobre cómo añadir la la clasificación de textos a una aplicación existente, consulte [Actualización y modificación de su aplicación](#modify_applications).

## Visión general de la clasificación de textos

La *clasificación de textos* es la tarea de aprendizaje automático que consiste en asignar un conjunto de categorías predefinidas a un texto abierto. Un modelo de clasificación de texto se entrena en un corpus de texto en lenguaje natural, en el que las palabras o frases se clasifican manualmente.

El modelo entrenado recibe texto como entrada e intenta categorizar el texto de acuerdo con el conjunto de clases conocidas para cuya clasificación fue entrenado. Por ejemplo, los modelos de este ejemplo aceptan un fragmento de texto y determinan si el sentimiento del texto es positivo o negativo. Para cada fragmento de texto, el modelo de clasificación de texto emite una puntuación que indica la confianza en que el texto se clasifique correctamente como positivo o negativo.

Para saber más sobre cómo se generan los modelos de este tutorial, consulte el tutorial [Clasificación de texto con Model Maker de TensorFlow Lite](https://www.tensorflow.org/lite/models/modify/model_maker/text_classification).

## Modelos y conjunto de datos

Este tutorial usa modelos que fueron entrenados usando el conjunto de datos [SST-2](https://nlp.stanford.edu/sentiment/index.html) (Stanford Sentiment Treebank). SST-2 contiene 67.349 reseñas de películas para entrenar y 872 reseñas de películas para analizar, con cada reseña categorizada como positiva o negativa. Los modelos usados en esta app fueron entrenados usando la herramienta [Model Maker](https://www.tensorflow.org/lite/models/modify/model_maker/text_classification) de TensorFlow Lite.

La aplicación de ejemplo usa los siguientes modelos preentrenados:

- [Vector de palabras promedio](https://www.tensorflow.org/lite/inference_with_metadata/task_library/nl_classifier) (`NLClassifier`): El `NLClassifier` de la librería de tareas clasifica el texto de entrada en diferentes categorías y puede manejar la mayoría de los modelos de clasificación de texto.

- [MobileBERT](https://www.tensorflow.org/lite/inference_with_metadata/task_library/bert_nl_classifier) (`BertNLClassifier`): El `BertNLClassifier` de la librería de tareas es similar al NLClassifier pero está adaptado para casos que requieren tokenizaciones de Wordpiece y Sentencepiece fuera del grafo.

## Configurar y ejecutar la app del ejemplo

Para configurar la aplicación de clasificación de textos, descargue la app de ejemplo de [GitHub](https://github.com/tensorflow/examples/tree/master/lite/examples/text_classification/android) y ejecútela utilizando [Android Studio](https://developer.android.com/studio/).

### Requisitos del sistema

- **<a>Android Studio</a>** versión 2021.1.1 (Bumblebee) o superior.
- Android SDK versión 31 o superior
- Dispositivo Android con una versión mínima del sistema operativo de SDK 21 (Android 7.0 - Nougat) con el [modo de desarrollador](https://developer.android.com/studio/debug/dev-options) activado o un emulador de Android.

### Obtener el código del ejemplo

Cree una copia local del código de ejemplo. Usará este código para crear un proyecto en Android Studio y ejecutar la aplicación de ejemplo.

Para clonar y configurar el código de ejemplo:

1. Clone el repositorio git
    <pre class="devsite-click-to-copy">    git clone https://github.com/tensorflow/examples.git
        </pre>
2. Opcionalmente, configure su instancia git para usar sparse checkout, de forma que sólo tenga los archivos para la app de ejemplo de clasificación de texto:
    <pre class="devsite-click-to-copy">    cd examples
        git sparse-checkout init --cone
        git sparse-checkout set lite/examples/text_classification/android
        </pre>

### Importar y ejecutar el proyecto

Cree un proyecto a partir del código de ejemplo descargado, compile el proyecto y ejecútelo.

Para importar y generar el proyecto de código de ejemplo:

1. Inicie [Android Studio](https://developer.android.com/studio).
2. En Android Studio, seleccione **Archivo &gt; Nuevo &gt; Importar proyecto**.
3. Navegue hasta el directorio de código de ejemplo que contiene el archivo `build.gradle` (<code>.../examples/lite/examples/text_classification/android/build.gradle</code>) y seleccione ese directorio.
4. Si Android Studio solicita una Sincronización Gradle, seleccione OK.
5. Asegúrese de que su dispositivo Android está conectado a la computadora y de que el modo de desarrollador está activado. Haga clic en la flecha verde `Run`.

Si selecciona el directorio correcto, Android Studio crea un nuevo proyecto y lo genera. Este proceso puede tardar unos minutos, dependiendo de la velocidad de su computadora y de si ha usado Android Studio para otros proyectos. Cuando la compilación se completa, Android Studio muestra un mensaje `GENERACIÓN EXITOSA` en el panel de estado **Resultado de generación**.

Para ejecutar el proyecto:

1. Desde Android Studio, ejecute el proyecto seleccionando **Ejecutar &gt; Ejecutar...**.
2. Seleccione un dispositivo Android conectado (o emulador) para analizar la app.

### Usar la aplicación

![App de ejemplo de clasificación de textos en Android](../../../images/lite/android/text-classification-screenshot.png){: .attempt-right width="250px"}

Tras ejecutar el proyecto en Android Studio, la aplicación se abre automáticamente en el dispositivo conectado o en el emulador del dispositivo.

Para usar el clasificador de texto:

1. Introduzca un fragmento de texto en el cuadro de texto.
2. En el menú desplegable **Delegado**, seleccione `CPU` o `NNAPI`.
3. Especifique un modelo seleccionando `AverageWordVec` o `MobileBERT`.
4. Seleccione **Clasificar**.

La aplicación emite una puntuación *positiva* y una *negativa*. Estas dos puntuaciones sumarán 1 y miden la probabilidad de que el sentimiento del texto de entrada sea positivo o negativo. Un número más alto denota un mayor nivel de confianza.

Ya dispone de una aplicación de clasificación de textos en funcionamiento. Utilice las siguientes secciones para comprender mejor cómo funciona la aplicación de ejemplo y cómo implementar las funciones de clasificación de textos en sus aplicaciones de producción:

- [Cómo funciona la aplicación](#how_it_works): Un recorrido por la estructura y los archivos clave de la aplicación de ejemplo.

- [Modifique su aplicación](#modify_applications): Instrucciones para añadir la clasificación de textos a una aplicación existente.

## Cómo funciona la aplicación {:#how_it_works}

La aplicación usa el paquete de [Librería de tareas para el lenguaje natural (NL)](../../inference_with_metadata/task_library/overview#supported_tasks) para implementar los modelos de clasificación de texto. Los dos modelos, Promedio de vectores de palabras y MobileBERT, se usaron para el entrenamiento mediante [Model Maker](https://www.tensorflow.org/lite/models/modify/model_maker/text_classification) de TensorFlow Lite. La aplicación se ejecuta de forma predeterminada en la CPU, con la opción de aceleración por hardware utilizando el delegado NNAPI.

Los siguientes archivos y directorios contienen el código crucial de esta aplicación de clasificación de textos:

- [TextClassificationHelper.kt](https://github.com/tensorflow/examples/blob/master/lite/examples/text_classification/android/app/src/main/java/org/tensorflow/lite/examples/textclassification/TextClassificationHelper.kt): Inicializa el clasificador de texto y se encarga de la selección del modelo y del delegado.
- [MainActivity.kt](https://github.com/tensorflow/examples/blob/master/lite/examples/text_classification/android/app/src/main/java/org/tensorflow/lite/examples/textclassification/MainActivity.kt): Implementa la aplicación, incluida la llamada a `TextClassificationHelper` y `ResultsAdapter`.
- [ResultsAdapter.kt](https://github.com/tensorflow/examples/blob/master/lite/examples/text_classification/android/app/src/main/java/org/tensorflow/lite/examples/textclassification/ResultsAdapter.kt): Maneja y formatea los resultados.

## Modificar su aplicación {:#modify_applications}

En las siguientes secciones se explican los pasos clave para modificar su propia app Android para que ejecute el modelo mostrado en la app de ejemplo. Estas instrucciones usan la app de ejemplo como punto de referencia. Los cambios específicos necesarios para su propia app pueden variar con respecto a la app de ejemplo.

### Abrir o crear un proyecto Android

Necesita un proyecto de desarrollo Android en Android Studio para seguir el resto de estas instrucciones. Siga las siguientes instrucciones para abrir un proyecto existente o crear uno nuevo.

Para abrir un proyecto de desarrollo Android existente:

- En Android Studio, seleccione *Archivo &gt; Abrir* y seleccione un proyecto existente.

Para crear un proyecto básico de desarrollo Android:

- Siga las instrucciones de Android Studio para [Crear un proyecto básico](https://developer.android.com/studio/projects/create-project).

Para saber más sobre cómo usar Android Studio, consulte la [Documentación de Android Studio](https://developer.android.com/studio/intro).

### Añadir las dependencias del proyecto

En su propia aplicación, debe añadir dependencias específicas del proyecto para ejecutar los modelos de aprendizaje automático de TensorFlow Lite, y acceder a las funciones de utilidad que convierten datos como cadenas de texto, en un formato de datos de tensor que puede ser procesado por el modelo que está usando.

Las siguientes instrucciones explican cómo añadir las dependencias necesarias del proyecto a su propio proyecto de app para Android.

Para añadir dependencias de módulos:

1. En el módulo que usa TensorFlow Lite, actualice el archivo `build.gradle` del módulo para incluir las siguientes dependencias.

    En la aplicación de ejemplo, las dependencias se encuentran en [app/build.gradle](https://github.com/tensorflow/examples/blob/master/lite/examples/text_classification/android/app/build.gradle):

    ```
    dependencies {
      ...
      implementation 'org.tensorflow:tensorflow-lite-task-text:0.4.0'
    }
    ```

    El proyecto debe incluir la librería de tareas de texto (`tensorflow-lite-task-text`).

    Si desea modificar esta app para ejecutarla en una unidad de procesamiento gráfico (GPU), la librería GPU (`tensorflow-lite-gpu-delegate-plugin`) ofrece la infraestructura para ejecutar la app en GPU, y Delegate (`tensorflow-lite-gpu`) provee la lista de compatibilidad. Ejecutar esta app en la GPU está fuera del alcance de este tutorial.

2. En Android Studio, sincronice las dependencias del proyecto seleccionando: **Archivo &gt; Sincronizar proyecto con archivos Gradle**.

### Inicializar los modelos ML {:#initialize_models}

En su app para Android, debe inicializar el modelo de aprendizaje automático TensorFlow Lite con parámetros antes de ejecutar predicciones con el modelo.

Un modelo TensorFlow Lite se almacena como un archivo `*.tflite`. El archivo del modelo contiene la lógica de predicción y normalmente incluye [metadatos](../../models/convert/metadata) sobre cómo interpretar los resultados de la predicción, como los nombres de las clases de predicción. Normalmente, los archivos de modelo se almacenan en el directorio `src/main/assets` de su proyecto de desarrollo, como en el ejemplo de código:

- `<project>/src/main/assets/mobilebert.tflite`
- `<project>/src/main/assets/wordvec.tflite`

Nota: La app de ejemplo usa un archivo `[download_model.gradle](https://github.com/tensorflow/examples/blob/master/lite/examples/text_classification/android/app/download_model.gradle)` para descargar los modelos [Average Word Vector](https://www.tensorflow.org/lite/inference_with_metadata/task_library/nl_classifier) y [MobileBERT](https://www.tensorflow.org/lite/inference_with_metadata/task_library/bert_nl_classifier) en tiempo de compilación. Este método no es necesario ni recomendable para una app de producción.

Por comodidad y legibilidad del código, el ejemplo declara un objeto complementario que define los ajustes del modelo.

Para inicializar el modelo en su app:

1. Cree un objeto complementario para definir los ajustes del modelo. En la aplicación de ejemplo, este objeto se encuentra en [TextClassificationHelper.kt](https://github.com/tensorflow/examples/blob/master/lite/examples/text_classification/android/app/src/main/java/org/tensorflow/lite/examples/textclassification/TextClassificationHelper.kt):

    ```
    companion object {
      const val DELEGATE_CPU = 0
      const val DELEGATE_NNAPI = 1
      const val WORD_VEC = "wordvec.tflite"
      const val MOBILEBERT = "mobilebert.tflite"
    }
    ```

2. Cree los ajustes para el modelo generando un objeto clasificador y construya un objeto TensorFlow Lite usando `BertNLClassifier` o `NLClassifier`.

    En la aplicación de ejemplo, se encuentra en la función `initClassifier` dentro de [TextClassificationHelper.kt](https://github.com/tensorflow/examples/blob/master/lite/examples/text_classification/android/app/src/main/java/org/tensorflow/lite/examples/textclassification/TextClassificationHelper.kt):

    ```
    fun initClassifier() {
      ...
      if( currentModel == MOBILEBERT ) {
        ...
        bertClassifier = BertNLClassifier.createFromFileAndOptions(
          context,
          MOBILEBERT,
          options)
      } else if (currentModel == WORD_VEC) {
          ...
          nlClassifier = NLClassifier.createFromFileAndOptions(
            context,
            WORD_VEC,
            options)
      }
    }
    ```

    Nota: La mayoría de las apps de producción que utilizan la clasificación de texto usarán `BertNLClassifier` o `NLClassifier`. - no ambos.

### Habilitar la aceleración por hardware (opcional) {:#hardware_acceleration}

Al inicializar un modelo TensorFlow Lite en su app, debe considerar la posibilidad de usar funciones de aceleración de hardware para acelerar los cálculos de predicción del modelo. Los [delegados](https://www.tensorflow.org/lite/performance/delegates) de TensorFlow Lite son módulos de software que aceleran la ejecución de modelos de aprendizaje automático usando hardware de procesamiento especializado en un dispositivo móvil, como unidades de procesamiento gráfico (GPU) o unidades de procesamiento tensorial (TPU).

Para habilitar la aceleración por hardware en su app:

1. Cree una variable para definir el delegado que usará la aplicación. En la aplicación de ejemplo, esta variable se encuentra al principio de [TextClassificationHelper.kt](https://github.com/tensorflow/examples/blob/master/lite/examples/text_classification/android/app/src/main/java/org/tensorflow/lite/examples/textclassification/TextClassificationHelper.kt):

    ```
    var currentDelegate: Int = 0
    ```

2. Cree un selector delegado. En la aplicación de ejemplo, el selector delegado se encuentra en la función `initClassifier` dentro de [TextClassificationHelper.kt](https://github.com/tensorflow/examples/blob/master/lite/examples/text_classification/android/app/src/main/java/org/tensorflow/lite/examples/textclassification/TextClassificationHelper.kt):

    ```
    val baseOptionsBuilder = BaseOptions.builder()
    when (currentDelegate) {
       DELEGATE_CPU -> {
           // Default
       }
       DELEGATE_NNAPI -> {
           baseOptionsBuilder.useNnapi()
       }
    }
    ```

Nota: Es posible modificar esta app para usar un delegado de GPU, pero esto requiere que el clasificador se cree en el mismo hilo que está usando el clasificador. Esto está fuera del alcance de este tutorial.

Usar delegados para ejecutar modelos TensorFlow Lite es recomendable, pero no obligatorio. Para saber más sobre cómo usar delegados con TensorFlow Lite, consulte [Delegados de TensorFlow Lite](https://www.tensorflow.org/lite/performance/delegates).

### Preparar los datos para el modelo

En su app para Android, su código ofrece datos al modelo para que los interprete transformando los datos existentes, como el texto puro, en un formato de datos [Tensor](https://www.tensorflow.org/lite/api_docs/java/org/tensorflow/lite/Tensor) que pueda ser procesado por su modelo. Los datos en un Tensor que pase a un modelo deben tener unas dimensiones específicas, o forma, que coincida con el formato de datos usado para entrenar al modelo.

Esta app de clasificación de textos acepta una [cadena](https://developer.android.com/reference/java/lang/String.html) como entrada, y los modelos se entrenan exclusivamente en un corpus de lengua inglesa. Los caracteres especiales y las palabras no inglesas se ignoran durante la inferencia.

Para aportar datos de texto al modelo:

1. Asegúrese de que la función `initClassifier` contiene el código para el delegado y los modelos, como se explica en las secciones [Inicializar los modelos ML](#initialize_models) y [Habilitar la aceleración por hardware](#hardware_acceleration).

2. Use el bloque `init` para llamar a la función `initClassifier`. En la aplicación de ejemplo, el `init` se encuentra en [TextClassificationHelper.kt](https://github.com/tensorflow/examples/blob/master/lite/examples/text_classification/android/app/src/main/java/org/tensorflow/lite/examples/textclassification/TextClassificationHelper.kt):

    ```
    init {
      initClassifier()
    }
    ```

### Ejecutar predicciones

En su app para Android, una vez que haya inicializado un objeto [BertNLClassifier](https://www.tensorflow.org/lite/api_docs/java/org/tensorflow/lite/task/text/nlclassifier/BertNLClassifier) o [NLClassifier](https://www.tensorflow.org/lite/api_docs/java/org/tensorflow/lite/task/text/nlclassifier/NLClassifier), puede empezar a introducir texto para que el modelo lo categorice como "positivo" o "negativo".

Para ejecutar predicciones:

1. Cree una función `classify`, que use el clasificador seleccionado (`currentModel`) y mida el tiempo empleado en clasificar el texto de entrada (`inferenceTime`). En la aplicación de ejemplo, la función `classify` se encuentra en [TextClassificationHelper.kt](https://github.com/tensorflow/examples/blob/master/lite/examples/text_classification/android/app/src/main/java/org/tensorflow/lite/examples/textclassification/TextClassificationHelper.kt):

    ```
    fun classify(text: String) {
      executor = ScheduledThreadPoolExecutor(1)

      executor.execute {
        val results: List<Category>
        // inferenceTime is the amount of time, in milliseconds, that it takes to
        // classify the input text.
        var inferenceTime = SystemClock.uptimeMillis()

        // Use the appropriate classifier based on the selected model
        if(currentModel == MOBILEBERT) {
          results = bertClassifier.classify(text)
        } else {
          results = nlClassifier.classify(text)
        }

        inferenceTime = SystemClock.uptimeMillis() - inferenceTime

        listener.onResult(results, inferenceTime)
      }
    }
    ```

2. Pasa los resultados de `classify` al objeto receptor de eventos.

    ```
    fun classify(text: String) {
      ...
      listener.onResult(results, inferenceTime)
    }
    ```

### Manejar la salida del modelo

Tras introducir una línea de texto, el modelo produce unas puntuaciones de predicción, expresadas como Float, entre 0 y 1 para las categorías "positivo" y "negativo".

To get the prediction results from the model:

1. Cree una función `onResult` para que el objeto receptor gestione la salida. En la aplicación de ejemplo, el objeto receptor se encuentra en [MainActivity.kt](https://github.com/tensorflow/examples/blob/master/lite/examples/text_classification/android/app/src/main/java/org/tensorflow/lite/examples/textclassification/MainActivity.kt).

    ```
    private val listener = object : TextClassificationHelper.TextResultsListener {
      override fun onResult(results: List<Category>, inferenceTime: Long) {
        runOnUiThread {
          activityMainBinding.bottomSheetLayout.inferenceTimeVal.text =
            String.format("%d ms", inferenceTime)

          adapter.resultsList = results.sortedByDescending {
            it.score
          }

          adapter.notifyDataSetChanged()
        }
      }
      ...
    }
    ```

2. Añada una función `onError` al objeto receptor para gestionar los errores:

    ```
      private val listener = object : TextClassificationHelper.TextResultsListener {
        ...
        override fun onError(error: String) {
          Toast.makeText(this@MainActivity, error, Toast.LENGTH_SHORT).show()
        }
      }
    ```

Una vez que el modelo ha devuelto un conjunto de resultados de predicción, su aplicación puede actuar sobre esas predicciones presentando el resultado a su usuario o ejecutando lógica adicional. La aplicación de ejemplo muestra las puntuaciones de las predicciones en la interfaz de usuario.

## Siguientes pasos

- Entrene e implemente los modelos desde cero con el tutorial [Clasificación de texto con Model Maker de TensorFlow Lite](https://www.tensorflow.org/lite/models/modify/model_maker/text_classification).
- Explore más [herramientas de procesamiento de texto para TensorFlow](https://www.tensorflow.org/text).
- Descargue otros modelos BERT en [TensorFlow Hub](https://tfhub.dev/google/collections/bert/1).
- Explore varios usos de TensorFlow Lite en los [ejemplos](../../examples).
- Aprenda más sobre cómo usar modelos de aprendizaje automático con TensorFlow Lite en la sección [Modelos](../../models).
- Aprenda más sobre cómo implementar el aprendizaje automático en su Aplicación Móvil en la [Guía para desarrolladores de TensorFlow Lite](../../guide).
