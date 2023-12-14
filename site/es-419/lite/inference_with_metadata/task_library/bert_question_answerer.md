# Integrar el contestador de preguntas BERT

La API `BertQuestionAnswerer` de la librería de tareas carga un modelo Bert y responde a las preguntas basándose en el contenido de un pasaje determinado. Para más información, consulte la documentación del modelo Pregunta-Respuesta <a href="../../examples/bert_qa/overview">aquí</a>.

## Características principales de la API BertQuestionAnswerer

- Toma dos entradas de texto como pregunta y contexto y emite una lista de posibles respuestas.

- Realiza tokenizaciones fuera del grafo de Wordpiece o Sentencepiece sobre el texto de entrada.

## Modelos BertQuestionAnswerer compatibles

Los siguientes modelos son compatibles con la API `BertNLClassifier`.

- Modelos creados por [Model Maker de TensorFlow Lite para BERT Question Answer](https://www.tensorflow.org/lite/models/modify/model_maker/question_answer).

- Los modelos BERT [prefiltrados en TensorFlow Hub](https://tfhub.dev/tensorflow/collections/lite/task-library/bert-question-answerer/1).

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

Nota: a partir de la versión 4.1 del plugin Gradle para Android, .tflite se añadirá a la lista noCompress de forma predeterminada y el aaptOptions anterior ya no será necesario.

### Paso 2: Ejecutar la inferencia usando la API

```java
// Initialization
BertQuestionAnswererOptions options =
    BertQuestionAnswererOptions.builder()
        .setBaseOptions(BaseOptions.builder().setNumThreads(4).build())
        .build();
BertQuestionAnswerer answerer =
    BertQuestionAnswerer.createFromFileAndOptions(
        androidContext, modelFile, options);

// Run inference
List<QaAnswer> answers = answerer.answer(contextOfTheQuestion, questionToAsk);
```

Consulte el [código fuente](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/java/src/java/org/tensorflow/lite/task/text/qa/BertQuestionAnswerer.java) para más detalles.

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
let mobileBertAnswerer = TFLBertQuestionAnswerer.questionAnswerer(
      modelPath: mobileBertModelPath)

// Run inference
let answers = mobileBertAnswerer.answer(
      context: context, question: question)
```

Consulte el [código fuente](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/ios/task/text/qa/Sources/TFLBertQuestionAnswerer.h) para más detalles.

## Ejecutar la inferencia en C++

```c++
// Initialization
BertQuestionAnswererOptions options;
options.mutable_base_options()->mutable_model_file()->set_file_name(model_path);
std::unique_ptr<BertQuestionAnswerer> answerer = BertQuestionAnswerer::CreateFromOptions(options).value();

// Run inference with your inputs, `context_of_question` and `question_to_ask`.
std::vector<QaAnswer> positive_results = answerer->Answer(context_of_question, question_to_ask);
```

Consulte el [código fuente](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/cc/task/text/bert_question_answerer.h) para más detalles.

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
answerer = text.BertQuestionAnswerer.create_from_file(model_path)

# Run inference
bert_qa_result = answerer.answer(context, question)
```

Consulte el [código fuente](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/python/task/text/bert_question_answerer.py) para ver más opciones para configurar `BertQuestionAnswerer`.

## Resultados de ejemplo

Aquí tiene un ejemplo de los resultados de la respuesta del [modelo ALBERT](https://tfhub.dev/tensorflow/lite-model/albert_lite_base/squadv1/1).

Contexto: "La selva amazónica, jungla amazónica, o Amazonia, es un bosque tropical húmedo de hoja ancha del bioma amazónico que cubre la mayor parte de la cuenca del Amazonas en Sudamérica. Esta cuenca abarca 7,000,000 km2 (2,700,000 millas cuadradas), de los cuales 5,500,000 km2 (2,100,000 millas cuadradas) están cubiertos por la selva. Esta región incluye territorio perteneciente a nueve naciones".

Pregunta: "¿Dónde está la selva amazónica?"

Respuestas:

```
answer[0]:  'South America.'
logit: 1.84847, start_index: 39, end_index: 40
answer[1]:  'most of the Amazon basin of South America.'
logit: 1.2921, start_index: 34, end_index: 40
answer[2]:  'the Amazon basin of South America.'
logit: -0.0959535, start_index: 36, end_index: 40
answer[3]:  'the Amazon biome that covers most of the Amazon basin of South America.'
logit: -0.498558, start_index: 28, end_index: 40
answer[4]:  'Amazon basin of South America.'
logit: -0.774266, start_index: 37, end_index: 40

```

Pruebe la sencilla herramienta demo [CLI para BertQuestionAnswerer](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/examples/task/text/desktop/README.md#bert-question-answerer) con su propio modelo y datos de prueba.

## Requisitos de compatibilidad del modelo

La API `BertQuestionAnswerer` espera un modelo TFLite con [Metadatos del modelo TFLite](../../models/convert/metadata) como requisito.

Los metadatos deben cumplir los siguientes requisitos:

- input_process_units para el tokenizador Wordpiece/Sentencepiece

- 3 tensores de entrada con los nombres "ids", "mask" y "segment_ids" para la salida del tokenizador

- 2 tensores de salida con nombres "end_logits" y "start_logits" para indicar la posición relativa de la respuesta en el contexto
