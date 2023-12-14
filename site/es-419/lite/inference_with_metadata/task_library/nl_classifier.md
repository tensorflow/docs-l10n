# Integrar el clasificador de lenguaje natural

La API `NLClassifier` de la librería de tareas clasifica el texto de entrada en diferentes categorías, y es una API versátil y configurable que puede manejar la mayoría de los modelos de clasificación de texto.

## Características principales de la API NLClassifier

- Toma una única cadena como entrada, realiza la clasificación con la cadena y emite pares &lt;Etiqueta, Puntuación&gt; como resultados de la clasificación.

- Tokenización Regex opcional disponible para el texto de entrada.

- Configurable para adaptar diferentes modelos de clasificación.

## Modelos de NLClassifier compatibles

Se garantiza que los siguientes modelos son compatibles con la API `NLClassifier`.

- El modelo de <a href="../../examples/text_classification/overview">clasificación del sentimiento de las críticas de cine</a>.

- Modelos con la especificación `average_word_vec` creados por [Model Maker de TensorFlow Lite para la clasificación de textos](https://www.tensorflow.org/lite/models/modify/model_maker/text_classification).

- Modelos personalizados que cumplen los [requisitos de compatibilidad de modelos](#model-compatibility-requirements).

## Ejecutar la inferencia en Java

Consulte la [app de referencia de clasificación de texto](https://github.com/tensorflow/examples/blob/master/lite/examples/text_classification/android/lib_task_api/src/main/java/org/tensorflow/lite/examples/textclassification/client/TextClassificationClient.java) para ver un ejemplo de cómo usar `NLClassifier` en una app de Android.

### Paso 1: Importar la dependencia de Gradle y otras configuraciones

Copie el archivo del modelo `.tflite` en el directorio de activos del módulo Android en el que se ejecutará el modelo. Especifique que el archivo no debe comprimirse y añada la librería TensorFlow Lite al archivo `build.gradle` del módulo:

```java
android {
    // Other settings

    // Specify tflite file should not be compressed for the app apk
    aaptOptions {
        noCompress "tflite"
    }

}

dependencies {
    // Other dependencies

    // Import the Task Vision Library dependency (NNAPI is included)
    implementation 'org.tensorflow:tensorflow-lite-task-text:0.4.4'
    // Import the GPU delegate plugin Library for GPU inference
    implementation 'org.tensorflow:tensorflow-lite-gpu-delegate-plugin:0.4.4'
}
```

Nota: a partir de la versión 4.1 del plugin Gradle para Android, .tflite se añadirá a la lista noCompress de forma predeterminada y el aaptOptions anterior ya no será necesario.

### Paso 2: Ejecutar la inferencia usando la API

```java
// Initialization, use NLClassifierOptions to configure input and output tensors
NLClassifierOptions options =
    NLClassifierOptions.builder()
        .setBaseOptions(BaseOptions.builder().useGpu().build())
        .setInputTensorName(INPUT_TENSOR_NAME)
        .setOutputScoreTensorName(OUTPUT_SCORE_TENSOR_NAME)
        .build();
NLClassifier classifier =
    NLClassifier.createFromFileAndOptions(context, modelFile, options);

// Run inference
List<Category> results = classifier.classify(input);
```

Consulte el [código fuente](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/java/src/java/org/tensorflow/lite/task/text/nlclassifier/NLClassifier.java) para más opciones de configuración de `NLClassifier`.

## Ejecutar la inferencia en Swift

### Paso 1: Importar CocoaPods

Añada el pod TensorFlowLiteTaskText en Podfile

```
target 'MySwiftAppWithTaskAPI' do
  use_frameworks!
  pod 'TensorFlowLiteTaskText', '~> 0.4.4'
end
```

### Paso 2: Ejecutar la inferencia usando la API

```swift
// Initialization
var modelOptions:TFLNLClassifierOptions = TFLNLClassifierOptions()
modelOptions.inputTensorName = inputTensorName
modelOptions.outputScoreTensorName = outputScoreTensorName
let nlClassifier = TFLNLClassifier.nlClassifier(
      modelPath: modelPath,
      options: modelOptions)

// Run inference
let categories = nlClassifier.classify(text: input)
```

Consulte el [código fuente](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/ios/task/text/nlclassifier/Sources/TFLNLClassifier.h) para más detalles.

## Ejecutar la inferencia en C++

```c++
// Initialization
NLClassifierOptions options;
options.mutable_base_options()->mutable_model_file()->set_file_name(model_path);
std::unique_ptr<NLClassifier> classifier = NLClassifier::CreateFromOptions(options).value();

// Run inference with your input, `input_text`.
std::vector<core::Category> categories = classifier->Classify(input_text);
```

Consulte el [código fuente](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/cc/task/text/nlclassifier/nl_classifier.h) para más detalles.

## Ejecutar la inferencia en Python

### Step 1: Install the pip package

```
pip install tflite-support
```

### Paso 2: Usar el modelo

```python
# Imports
from tflite_support.task import text

# Initialization
classifier = text.NLClassifier.create_from_file(model_path)

# Run inference
text_classification_result = classifier.classify(text)
```

Consulte el [código fuente](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/python/task/text/nl_classifier.py) para más opciones de configuración de `NLClassifier`.

## Resultados de ejemplo

Este es un ejemplo de los resultados de la clasificación del [modelo de crítica de cine](https://www.tensorflow.org/lite/examples/text_classification/overview).

Entrada: "Qué pérdida de tiempo".

Salida

```
category[0]: 'Negative' : '0.81313'
category[1]: 'Positive' : '0.18687'
```

Pruebe la sencilla herramienta demo [CLI para NLClassifier](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/examples/task/text/desktop/README.md#nlclassifier) con su propio modelo y datos de prueba.

## Requisitos de compatibilidad del modelo

Según el caso de uso, la API `NLClassifier` puede cargar un modelo TFLite con o sin [Metadatos del modelo TFLite](../../models/convert/metadata). Vea ejemplos de creación de metadatos para clasificadores de lenguaje natural usando la [API escritora de metadatos de TensorFlow Lite](../../models/convert/metadata_writer_tutorial.ipynb#nl_classifiers).

Los modelos compatibles deben cumplir los siguientes requisitos:

- Tensor de entrada: (kTfLiteString/kTfLiteInt32)

    - La entrada del modelo debe ser un tensor kTfLiteString de cadena de entrada sin procesar o un tensor kTfLiteInt32 para índices tokenizados regex de cadena de entrada sin procesar.
    - Si el tipo de entrada es kTfLiteString, no se requieren [Metadatos](../../models/convert/metadata) para el modelo.
    - Si el tipo de entrada es kTfLiteInt32, es necesario configurar un `RegexTokenizer` en los [Metadatos](https://www.tensorflow.org/lite/models/convert/metadata_writer_tutorial#natural_language_classifiers) del tensor de entrada.

- Tensor de puntuaciones de salida: (kTfLiteUInt8/kTfLiteInt8/kTfLiteInt16/kTfLiteFloat32/kTfLiteFloat64)

    - Tensor de salida obligatorio para las puntuaciones de cada categoría clasificada.

    - Si el tipo es uno de los tipos Int, lo decuantiza a double/float a las plataformas correspondientes

    - Puede tener un archivo opcional asociado en los [Metadatos](../../models/convert/metadata) correspondientes del tensor de salida para las etiquetas de las categorías, el archivo debe ser un archivo de texto sin formato con una etiqueta por línea, y el número de etiquetas debe coincidir con el número de categorías como salidas del modelo. Consulte el [archivo de etiquetas de ejemplo](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/metadata/python/tests/testdata/nl_classifier/labels.txt).

- Tensor de etiquetas de salida: (kTfLiteString/kTfLiteInt32)

    - Tensor de salida opcional para la etiqueta de cada categoría, debe tener la misma longitud que el tensor de puntuaciones de salida. Si este tensor no está presente, la API usa índices de puntuaciones como nombres de clase.

    - Se ignorará si el archivo de etiquetas asociado está presente en los metadatos del tensor de puntuaciones de salida.
