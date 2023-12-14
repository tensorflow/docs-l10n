# Integrar el clasificador de lenguaje natural BERT

La API `BertNLClassifier` de la librería de tareas es muy similar al `NLClassifier` que clasifica el texto de entrada en diferentes categorías, salvo que esta API está especialmente adaptada para los modelos relacionados con Bert que requieren tokenizaciones Wordpiece y Sentencepiece fuera del modelo TFLite.

## Características principales de la API BertNLClassifier

- Toma una única cadena como entrada, realiza la clasificación con la cadena y emite pares &lt;Etiqueta, Puntuación&gt; como resultados de la clasificación.

- Realiza tokenizaciones fuera de grafo [Wordpiece](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/cc/text/tokenizers/bert_tokenizer.h) o [Sentencepiece](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/cc/text/tokenizers/sentencepiece_tokenizer.h) sobre el texto de entrada.

## Modelos BertNLClassifier compatibles

Los siguientes modelos son compatibles con la API `BertNLClassifier`.

- Modelos Bert creados por [Model Maker de TensorFlow Lite para la clasificación de textos](https://www.tensorflow.org/lite/models/modify/model_maker/text_classification).

- Modelos personalizados que cumplen los [requisitos de compatibilidad de modelos](#model-compatibility-requirements).

## Ejecutar la inferencia en Java

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

    // Import the Task Text Library dependency (NNAPI is included)
    implementation 'org.tensorflow:tensorflow-lite-task-text:0.4.4'
}
```

Nota: a partir de la versión 4.1 del plugin Gradle para Android, .tflite se añadirá a la lista noCompress de forma predeterminada y el aaptOptions anterior ya no es necesario.

### Paso 2: Ejecutar la inferencia usando la API

```java
// Initialization
BertNLClassifierOptions options =
    BertNLClassifierOptions.builder()
        .setBaseOptions(BaseOptions.builder().setNumThreads(4).build())
        .build();
BertNLClassifier classifier =
    BertNLClassifier.createFromFileAndOptions(context, modelFile, options);

// Run inference
List<Category> results = classifier.classify(input);
```

Consulte el [código fuente](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/java/src/java/org/tensorflow/lite/task/text/nlclassifier/BertNLClassifier.java) para más detalles.

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
let bertNLClassifier = TFLBertNLClassifier.bertNLClassifier(
      modelPath: bertModelPath)

// Run inference
let categories = bertNLClassifier.classify(text: input)
```

Consulte el [código fuente](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/ios/task/text/nlclassifier/Sources/TFLBertNLClassifier.h) para más detalles.

## Ejecutar la inferencia en C++

```c++
// Initialization
BertNLClassifierOptions options;
options.mutable_base_options()->mutable_model_file()->set_file_name(model_path);
std::unique_ptr<BertNLClassifier> classifier = BertNLClassifier::CreateFromOptions(options).value();

// Run inference with your input, `input_text`.
std::vector<core::Category> categories = classifier->Classify(input_text);
```

Consulte el [código fuente](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/cc/task/text/bert_nl_classifier.h) para más detalles.

## Ejecutar la inferencia en Python

### Paso 1: Instalar el paquete pip

```
pip install tflite-support
```

### Paso 2: Usar el modelo

```python
# Imports
from tflite_support.task import text

# Initialization
classifier = text.BertNLClassifier.create_from_file(model_path)

# Run inference
text_classification_result = classifier.classify(text)
```

Consulte el [código fuente](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/python/task/text/bert_nl_classifier.py) para ver más opciones para configurar `BertNLClassifier`.

## Resultados de ejemplo

Este es un ejemplo de los resultados de la clasificación de críticas de películas usando el modelo [MobileBert](https://www.tensorflow.org/lite/models/modify/model_maker/text_classification) de Model Maker.

Entrada: "es una aventura encantadora y a menudo conmovedora"

Salida:

```
category[0]: 'negative' : '0.00006'
category[1]: 'positive' : '0.99994'
```

Pruebe la sencilla herramienta demo [CLI para BertNLClassifier](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/examples/task/text/desktop/README.md#bertnlclassifier) con su propio modelo y datos de prueba.

## Requisitos de compatibilidad del modelo

La API `BetNLClassifier` espera un modelo TFLite con [Metadatos del modelo TFLite](../../models/convert/metadata.md) como requisito.

Los metadatos deben cumplir los siguientes requisitos:

- input_process_units para el tokenizador Wordpiece/Sentencepiece

- 3 tensores de entrada con los nombres "ids", "mask" y "segment_ids" para la salida del tokenizador

- 1 tensor de salida de tipo float32, con un archivo de etiquetas opcionalmente adjunto. Si se adjunta un archivo de etiquetas, éste debe ser un archivo de texto sin formato con una etiqueta por línea y el número de etiquetas debe coincidir con el número de categorías como salidas del modelo.
