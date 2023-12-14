# Responder preguntas con Android

![App de ejemplo de respuesta a preguntas en Android](../../examples/bert_qa/images/screenshot.gif){: .attempt-right width="250px"}

Este tutorial le muestra cómo construir una aplicación Android usando TensorFlow Lite para dar respuesta a preguntas estructuradas en texto en lenguaje natural. La [aplicación de ejemplo](https://github.com/tensorflow/examples/tree/master/lite/examples/bert_qa/android) usa la API de *respondedor de preguntas de Bert* ([`BertQuestionAnswerer`](../../inference_with_metadata/task_library/bert_question_answerer)) dentro de la [Librería de tareas para lenguaje natural (NL)](../../inference_with_metadata/task_library/overview#supported_tasks) para habilitar modelos de aprendizaje automático de respuesta a preguntas. La aplicación está diseñada para un dispositivo Android físico, pero también puede ejecutarse en un emulador de dispositivo.

Si está actualizando un proyecto existente, puede usar la aplicación de ejemplo como referencia o plantilla. Si desea instrucciones sobre cómo añadir la respuesta a preguntas a una aplicación existente, consulte [Actualización y modificación de su aplicación](#modify_applications).

## Visión general sobre responder preguntas

*La respuesta a preguntas* es la tarea de aprendizaje automático que consiste en responder a preguntas formuladas en lenguaje natural. Un modelo de respuesta a preguntas entrenado recibe como entrada un pasaje de texto y una pregunta, e intenta responder a la pregunta basándose en su interpretación de la información contenida en el pasaje.

Se entrena un modelo de respuesta a preguntas en un conjunto de datos de respuesta a preguntas, que consiste en un conjunto de datos de Comprensión lectora junto con pares de pregunta-respuesta basados en diferentes segmentos de texto.

Para saber más sobre cómo se generan los modelos de este tutorial, consulte el tutorial [Respondedor de preguntas BERT con Model Maker de TensorFlow Lite](https://www.tensorflow.org/lite/models/modify/model_maker/question_answer).

## Modelos y conjunto de datos

La app de ejemplo usa el modelo BERT móvil Q&amp;A ([`mobilebert`](https://tfhub.dev/tensorflow/lite-model/mobilebert/1/metadata/1)), que es una versión más ligera y rápida de [BERT](https://arxiv.org/abs/1810.04805) (representaciones codificadoras bidireccionales a partir de transformadores, BERT). Para saber más sobre `mobilebert`, consulte el documento de investigación [MobileBERT: a Compact Task-Agnostic BERT for Resource-Limited Devices](https://arxiv.org/abs/2004.02984).

El modelo `mobilebert` se entrenó utilizando el conjunto de datos de respuestas a preguntas de Stanford ([SQuAD](https://rajpurkar.github.io/SQuAD-explorer/)), un conjunto de datos de Comprensión lectora formado por artículos de Wikipedia y un conjunto de pares pregunta-respuesta para cada artículo.

## Configurar y ejecutar la app del ejemplo

Para configurar la aplicación de respuesta a preguntas, descargue la app de ejemplo de [GitHub](https://github.com/tensorflow/examples/tree/master/lite/examples/bert_qa/android) y ejecútela usando [Android Studio](https://developer.android.com/studio/).

### Requisitos del sistema

- <a>Android Studio</a> versión 2021.1.1 (Bumblebee) o superior.
- Android SDK versión 31 o superior
- Dispositivo Android con una versión mínima del sistema operativo de SDK 21 (Android 7.0 - Nougat) con el [modo de desarrollador](https://developer.android.com/studio/debug/dev-options) activado o un emulador de Android.

### Obtener el código del ejemplo

Cree una copia local del código de ejemplo. Usará este código para crear un proyecto en Android Studio y ejecutar la aplicación de ejemplo.

Para clonar y configurar el código de ejemplo:

1. Clone el repositorio git
    <pre class="devsite-click-to-copy">    git clone https://github.com/tensorflow/examples.git
        </pre>
2. Opcionalmente, configure su instancia de git para usar sparse checkout, de forma que sólo tenga los archivos de la app de ejemplo de respuesta a preguntas:
    <pre class="devsite-click-to-copy">    cd examples
        git sparse-checkout init --cone
        git sparse-checkout set lite/examples/bert_qa/android
        </pre>

### Importar y ejecutar el proyecto

Cree un proyecto a partir del código de ejemplo descargado, compile el proyecto y ejecútelo.

Para importar y generar el proyecto de código de ejemplo:

1. Inicie [Android Studio](https://developer.android.com/studio).
2. En Android Studio, seleccione **Archivo &gt; Nuevo &gt; Importar proyecto**.
3. Navegue hasta el directorio de código de ejemplo que contiene el archivo build.gradle (`.../examples/lite/examples/bert_qa/android/build.gradle`) y seleccione ese directorio.
4. Si Android Studio solicita una Sincronización Gradle, seleccione OK.
5. Asegúrese de que su dispositivo Android está conectado a la computadora y de que el modo de desarrollador está activado. Haga clic en la flecha verde `Run`.

Si selecciona el directorio correcto, Android Studio crea un nuevo proyecto y lo genera. Este proceso puede tardar unos minutos, dependiendo de la velocidad de su computadora y de si ha usado Android Studio para otros proyectos. Cuando la compilación se completa, Android Studio muestra un mensaje `BUILD SUCCESSFUL` en el panel de estado **Resultado de la generación**.

Para ejecutar el proyecto:

1. Desde Android Studio, ejecute el proyecto seleccionando **Ejecutar &gt; Ejecutar...**.
2. Seleccione un dispositivo Android conectado (o emulador) para analizar la app.

### Usar la aplicación

Tras ejecutar el proyecto en Android Studio, la aplicación se abre automáticamente en el dispositivo conectado o en el emulador del dispositivo.

Para usar la app de ejemplo Respondedor de preguntas:

1. Seleccione un tema de la lista de materias.
2. Seleccione una pregunta sugerida o introduzca la suya propia en el cuadro de texto.
3. Pulse la flecha naranja para ejecutar el modelo.

La aplicación intenta identificar la respuesta a la pregunta a partir del texto del pasaje. Si el modelo detecta una respuesta dentro del pasaje, la aplicación resalta el alcance pertinente del texto para el usuario.

Ya dispone de una aplicación de respuesta a preguntas en funcionamiento. Utilice las siguientes secciones para comprender mejor cómo funciona la aplicación de ejemplo y cómo implementar las funciones de respuesta a preguntas en sus aplicaciones de producción:

- [Cómo funciona la aplicación](#how_it_works): Un recorrido por la estructura y los archivos clave de la aplicación de ejemplo.

- [Modifique su aplicación](#modify_applications): Instrucciones para añadir la respuesta a preguntas a una aplicación existente.

## Cómo funciona la aplicación {:#how_it_works}

La aplicación usa la API `BertQuestionAnswerer` que viene dentro del paquete [Librería de tareas para el lenguaje natural (NL)](../../inference_with_metadata/task_library/overview#supported_tasks). El modelo MobileBERT fue entrenado usando el [Model Maker](https://www.tensorflow.org/lite/models/modify/model_maker/question_answer) de TensorFlow Lite. La aplicación se ejecuta de forma predeterminada en la CPU, con la opción de aceleración por hardware utilizando la GPU o el delegado NNAPI.

Los siguientes archivos y directorios contienen el código crucial de esta aplicación:

- [BertQaHelper.kt](https://github.com/tensorflow/examples/blob/master/lite/examples/bert_qa/android/app/src/main/java/org/tensorflow/lite/examples/bertqa/BertQaHelper.kt): Inicializa el contestador de preguntas y se encarga de la selección del modelo y del delegado.
- [QaFragment.kt](https://github.com/tensorflow/examples/blob/master/lite/examples/bert_qa/android/app/src/main/java/org/tensorflow/lite/examples/bertqa/fragments/QaFragment.kt): Maneja y formatea los resultados.
- [MainActivity.kt](https://github.com/tensorflow/examples/blob/master/lite/examples/bert_qa/android/app/src/main/java/org/tensorflow/lite/examples/bertqa/MainActivity.kt): Proporciona la lógica organizativa de la app.

## Modifique su aplicación {:#modify_applications}

En las siguientes secciones se explican los pasos clave para modificar su propia app Android para que ejecute el modelo mostrado en la app de ejemplo. Estas instrucciones usan la app de ejemplo como punto de referencia. Los cambios específicos necesarios para su propia app pueden variar con respecto a la app de ejemplo.

### Abrir o crear un proyecto Android

Necesita un proyecto de desarrollo Android en Android Studio para seguir el resto de estas instrucciones. Siga las siguientes instrucciones para abrir un proyecto existente o crear uno nuevo.

Para abrir un proyecto de desarrollo Android existente:

- En Android Studio, seleccione *Archivo &gt; Abrir* y seleccione un proyecto existente.

Para crear un proyecto básico de desarrollo Android:

- Siga las instrucciones de Android Studio para [Crear un proyecto básico](https://developer.android.com/studio/projects/create-project).

Para saber más sobre cómo usar Android Studio, consulte la [Documentación de Android Studio](https://developer.android.com/studio/intro).

### Añadir las dependencias del proyecto

En su propia aplicación, añada dependencias específicas del proyecto para ejecutar los modelos de aprendizaje automático de TensorFlow Lite y acceder a las funciones de utilidad. Estas funciones convierten datos como cadenas en un formato de datos de tensor que puede ser procesado por el modelo. Las siguientes instrucciones explican cómo añadir las dependencias de proyecto y módulo necesarias a su propio proyecto de app para Android.

Para añadir dependencias de módulos:

1. En el módulo que usa TensorFlow Lite, actualice el archivo `build.gradle` del módulo para incluir las siguientes dependencias.

    En la aplicación de ejemplo, las dependencias se encuentran en [app/build.gradle](https://github.com/tensorflow/examples/blob/master/lite/examples/bert_qa/android/app/build.gradle):

    ```
    dependencies {
      ...
      // Import tensorflow library
      implementation 'org.tensorflow:tensorflow-lite-task-text:0.3.0'

      // Import the GPU delegate plugin Library for GPU inference
      implementation 'org.tensorflow:tensorflow-lite-gpu-delegate-plugin:0.4.0'
      implementation 'org.tensorflow:tensorflow-lite-gpu:2.9.0'
    }
    ```

    El proyecto debe incluir la librería de tareas de texto (`tensorflow-lite-task-text`).

    Si desea modificar esta app para ejecutarla en una unidad de procesamiento gráfico (GPU), la librería GPU (`tensorflow-lite-gpu-delegate-plugin`) ofrece la infraestructura para ejecutar la app en GPU, y Delegate (`tensorflow-lite-gpu`) provee la lista de compatibilidad.

2. En Android Studio, sincronice las dependencias del proyecto seleccionando: **Archivo &gt; Sincronizar proyecto con archivos Gradle**.

### Inicializar los modelos ML {:#initialize_models}

En su app para Android, debe inicializar el modelo de aprendizaje automático TensorFlow Lite con parámetros antes de ejecutar predicciones con el modelo.

Un modelo TensorFlow Lite se almacena como un archivo `*.tflite`. El archivo de modelo contiene la lógica de predicción y normalmente incluye [metadatos](../../models/convert/metadata) sobre cómo interpretar los resultados de la predicción. Normalmente, los archivos de modelo se almacenan en el directorio `src/main/assets` de su proyecto de desarrollo, como en el ejemplo de código:

- `<project>/src/main/assets/mobilebert_qa.tflite`

Nota: La app de ejemplo usa un archivo [`download_model.gradle`](https://github.com/tensorflow/examples/blob/master/lite/examples/bert_qa/android/app/download_models.gradle) para descargar el modelo [mobilebert_qa](https://www.tensorflow.org/lite/inference_with_metadata/task_library/bert_question_answerer) y el [pasaje de texto](https://storage.googleapis.com/download.tensorflow.org/models/tflite/bert_qa/contents_from_squad.json) en el momento de la generación. Este enfoque no es necesario para una app de producción.

Por comodidad y legibilidad del código, el ejemplo declara un objeto complementario que define los ajustes del modelo.

Para inicializar el modelo en su app:

1. Cree un objeto complementario para definir los ajustes del modelo. En la aplicación de ejemplo, este objeto se encuentra en [BertQaHelper.kt](https://github.com/tensorflow/examples/blob/master/lite/examples/bert_qa/android/app/src/main/java/org/tensorflow/lite/examples/bertqa/BertQaHelper.kt#L100-L106):

    ```
    companion object {
        private const val BERT_QA_MODEL = "mobilebert.tflite"
        private const val TAG = "BertQaHelper"
        const val DELEGATE_CPU = 0
        const val DELEGATE_GPU = 1
        const val DELEGATE_NNAPI = 2
    }
    ```

2. Cree los ajustes para el modelo construyendo un objeto `BertQaHelper`, y construya un objeto TensorFlow Lite con `bertQuestionAnswerer`.

    En la aplicación de ejemplo, se encuentra en la función `setupBertQuestionAnswerer()` dentro de [BertQaHelper.kt](https://github.com/tensorflow/examples/blob/master/lite/examples/bert_qa/android/app/src/main/java/org/tensorflow/lite/examples/bertqa/BertQaHelper.kt#L41-L76):

    ```
    class BertQaHelper(
        ...
    ) {
        ...
        init {
            setupBertQuestionAnswerer()
        }

        fun clearBertQuestionAnswerer() {
            bertQuestionAnswerer = null
        }

        private fun setupBertQuestionAnswerer() {
            val baseOptionsBuilder = BaseOptions.builder().setNumThreads(numThreads)
            ...
            val options = BertQuestionAnswererOptions.builder()
                .setBaseOptions(baseOptionsBuilder.build())
                .build()

            try {
                bertQuestionAnswerer =
                    BertQuestionAnswerer.createFromFileAndOptions(context, BERT_QA_MODEL, options)
            } catch (e: IllegalStateException) {
                answererListener
                    ?.onError("Bert Question Answerer failed to initialize. See error logs for details")
                Log.e(TAG, "TFLite failed to load model with error: " + e.message)
            }
        }
        ...
        }
    ```

### Habilitar la aceleración por hardware (opcional) {:#hardware_acceleration}

Al inicializar un modelo TensorFlow Lite en su app, debe considerar la posibilidad de usar funciones de aceleración de hardware para acelerar los cálculos de predicción del modelo. Los [delegados](https://www.tensorflow.org/lite/performance/delegates) de TensorFlow Lite son módulos de software que aceleran la ejecución de modelos de aprendizaje automático usando hardware de procesamiento especializado en un dispositivo móvil, como unidades de procesamiento gráfico (GPU) o unidades de procesamiento tensorial (TPU).

Para habilitar la aceleración por hardware en su app:

1. Cree una variable para definir el delegado que usará la aplicación. En la aplicación de ejemplo, esta variable se encuentra al principio de [BertQaHelper.kt](https://github.com/tensorflow/examples/blob/master/lite/examples/bert_qa/android/app/src/main/java/org/tensorflow/lite/examples/bertqa/BertQaHelper.kt#L31):

    ```
    var currentDelegate: Int = 0
    ```

2. Cree un selector delegado. En la aplicación de ejemplo, el selector delegado se encuentra en la función `setupBertQuestionAnswerer` dentro de [BertQaHelper.kt](https://github.com/tensorflow/examples/blob/master/lite/examples/bert_qa/android/app/src/main/java/org/tensorflow/lite/examples/bertqa/BertQaHelper.kt#L48-L62):

    ```
    when (currentDelegate) {
        DELEGATE_CPU -> {
            // Default
        }
        DELEGATE_GPU -> {
            if (CompatibilityList().isDelegateSupportedOnThisDevice) {
                baseOptionsBuilder.useGpu()
            } else {
                answererListener?.onError("GPU is not supported on this device")
            }
        }
        DELEGATE_NNAPI -> {
            baseOptionsBuilder.useNnapi()
        }
    }
    ```

Usar delegados para ejecutar modelos TensorFlow Lite es recomendable, pero no obligatorio. Para saber más sobre cómo usar delegados con TensorFlow Lite, consulte [Delegados de TensorFlow Lite](https://www.tensorflow.org/lite/performance/delegates).

### Preparar los datos para el modelo

En su app para Android, su código ofrece datos al modelo para que los interprete transformando los datos existentes, como texto sin procesar, en un formato de datos [Tensor](https://www.tensorflow.org/lite/api_docs/java/org/tensorflow/lite/Tensor) que pueda ser procesado por su modelo. El tensor que usted pasa a un modelo debe tener unas dimensiones específicas, o una forma, que coincida con el formato de los datos usados para entrenar el modelo. Esta app para responder preguntas acepta [cadenas](https://developer.android.com/reference/java/lang/String.html) como entradas tanto para el pasaje de texto como para la pregunta. El modelo no reconoce los caracteres especiales ni las palabras que no estén en inglés.

Para aportar datos del pasaje de texto al modelo:

1. Use el objeto `LoadDataSetClient` para cargar los datos del pasaje de texto en la app. En la aplicación de ejemplo, se encuentra en [LoadDataSetClient.kt](https://github.com/tensorflow/examples/blob/master/lite/examples/bert_qa/android/app/src/main/java/org/tensorflow/lite/examples/bertqa/dataset/LoadDataSetClient.kt#L25-L45).

    ```
    fun loadJson(): DataSet? {
        var dataSet: DataSet? = null
        try {
            val inputStream: InputStream = context.assets.open(JSON_DIR)
            val bufferReader = inputStream.bufferedReader()
            val stringJson: String = bufferReader.use { it.readText() }
            val datasetType = object : TypeToken<DataSet>() {}.type
            dataSet = Gson().fromJson(stringJson, datasetType)
        } catch (e: IOException) {
            Log.e(TAG, e.message.toString())
        }
        return dataSet
    }
    ```

2. Use el objeto `DatasetFragment` para listar los títulos de cada pasaje de texto e iniciar la pantalla **Preguntas y Respuestas de TFL**. En la aplicación de ejemplo, esto se encuentra en [DatasetFragment.kt](https://github.com/tensorflow/examples/blob/master/lite/examples/bert_qa/android/app/src/main/java/org/tensorflow/lite/examples/bertqa/fragments/DatasetFragment.kt):

    ```
    class DatasetFragment : Fragment() {
        ...
        override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
            super.onViewCreated(view, savedInstanceState)
            val client = LoadDataSetClient(requireActivity())
            client.loadJson()?.let {
                titles = it.getTitles()
            }
            ...
        }
       ...
    }
    ```

3. Use la función `onCreateViewHolder` dentro del objeto `DatasetAdapter` para presentar los títulos de cada pasaje de texto. En la aplicación de ejemplo, esto se encuentra en [DatasetAdapter.kt](https://github.com/tensorflow/examples/blob/master/lite/examples/bert_qa/android/app/src/main/java/org/tensorflow/lite/examples/bertqa/fragments/DatasetAdapter.kt):

    ```
    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): ViewHolder {
        val binding = ItemDatasetBinding.inflate(
            LayoutInflater.from(parent.context),
            parent,
            false
        )
        return ViewHolder(binding)
    }
    ```

Para plantear preguntas de usuario al modelo:

1. Use el objeto `QaAdapter` para facilitar la pregunta al modelo. En la aplicación de ejemplo, se encuentra en [QaAdapter.kt](https://github.com/tensorflow/examples/blob/master/lite/examples/bert_qa/android/app/src/main/java/org/tensorflow/lite/examples/bertqa/fragments/QaAdapter.kt):

    ```
    class QaAdapter(private val question: List<String>, private val select: (Int) -> Unit) :
      RecyclerView.Adapter<QaAdapter.ViewHolder>() {

      inner class ViewHolder(private val binding: ItemQuestionBinding) :
          RecyclerView.ViewHolder(binding.root) {
          init {
              binding.tvQuestionSuggestion.setOnClickListener {
                  select.invoke(adapterPosition)
              }
          }

          fun bind(question: String) {
              binding.tvQuestionSuggestion.text = question
          }
      }
      ...
    }
    ```

### Ejecutar predicciones

En su app Android, una vez que haya inicializado un objeto [BertQuestionAnswerer](https://www.tensorflow.org/lite/api_docs/java/org/tensorflow/lite/task/text/qa/BertQuestionAnswerer), puede empezar a introducir preguntas en forma de texto en lenguaje natural al modelo. El modelo intentará identificar la respuesta dentro del pasaje de texto.

Para ejecutar predicciones:

1. Cree una función `answer`, que ejecute el modelo y mida el tiempo (`inferenceTime`) que se tarda en identificar la respuesta. En la aplicación de ejemplo, la función `answer` se encuentra en [BertQaHelper.kt](https://github.com/tensorflow/examples/blob/master/lite/examples/bert_qa/android/app/src/main/java/org/tensorflow/lite/examples/bertqa/BertQaHelper.kt#L78-L98):

    ```
    fun answer(contextOfQuestion: String, question: String) {
        if (bertQuestionAnswerer == null) {
            setupBertQuestionAnswerer()
        }

        var inferenceTime = SystemClock.uptimeMillis()

        val answers = bertQuestionAnswerer?.answer(contextOfQuestion, question)
        inferenceTime = SystemClock.uptimeMillis() - inferenceTime
        answererListener?.onResults(answers, inferenceTime)
    }
    ```

2. Pase los resultados de `answer` al objeto receptor.

    ```
    interface AnswererListener {
        fun onError(error: String)
        fun onResults(
            results: List<QaAnswer>?,
            inferenceTime: Long
        )
    }
    ```

### Manejar la salida del modelo

Tras introducir una pregunta, el modelo ofrece un máximo de cinco respuestas posibles dentro del pasaje.

Para obtener los resultados del modelo:

1. Cree una función `onResult` para que el objeto receptor gestione la salida. En la aplicación de ejemplo, el objeto receptor se encuentra en [BertQaHelper.kt](https://github.com/tensorflow/examples/blob/master/lite/examples/bert_qa/android/app/src/main/java/org/tensorflow/lite/examples/bertqa/BertQaHelper.kt#L92-L98).

    ```
    interface AnswererListener {
        fun onError(error: String)
        fun onResults(
            results: List<QaAnswer>?,
            inferenceTime: Long
        )
    }
    ```

2. Resalte las secciones del pasaje en función de los resultados. En la aplicación de ejemplo, esto se encuentra en [QaFragment.kt](https://github.com/tensorflow/examples/blob/master/lite/examples/bert_qa/android/app/src/main/java/org/tensorflow/lite/examples/bertqa/fragments/QaFragment.kt#L199-L208):

    ```
    override fun onResults(results: List<QaAnswer>?, inferenceTime: Long) {
        results?.first()?.let {
            highlightAnswer(it.text)
        }

        fragmentQaBinding.tvInferenceTime.text = String.format(
            requireActivity().getString(R.string.bottom_view_inference_time),
            inferenceTime
        )
    }
    ```

Una vez que el modelo ha devuelto una serie de resultados, su aplicación puede actuar sobre esas predicciones presentando el resultado a su usuario o ejecutando una lógica adicional.

## Siguientes pasos

- Entrene e implemente los modelos desde cero con el tutorial [Respuesta a preguntas con Model Maker de TensorFlow Lite](https://www.tensorflow.org/lite/models/modify/model_maker/question_answer).
- Explore más [herramientas de procesamiento de texto para TensorFlow](https://www.tensorflow.org/text).
- Descargue otros modelos BERT en [TensorFlow Hub](https://tfhub.dev/google/collections/bert/1).
- Explore varios usos de TensorFlow Lite en los [ejemplos](../../examples).
