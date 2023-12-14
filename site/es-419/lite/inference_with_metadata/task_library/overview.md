# Librería de tareas TensorFlow Lite

La librería de tareas de TensorFlow Lite contiene un conjunto de librerías específicas de tareas potentes y fáciles de usar para que los desarrolladores de apps creen experiencias de Aprendizaje automático con TensorFlow Lite. Provee interfaces de modelo listas para usar optimizadas para tareas populares de aprendizaje automático, como clasificación de imágenes, preguntas y respuestas, etc. Las interfaces de modelo están diseñadas específicamente para cada tarea con el fin de lograr el mejor rendimiento y usabilidad. La librería de tareas funciona en varias plataformas y es compatible con Java, C++ y Swift.

## Qué esperar de la librería de tareas

- **{nbsp}APIs pulcras y bien definidas utilizables por no expertos en aprendizaje automático** <br> La inferencia puede realizarse con sólo 5 líneas de código. Use las API potentes y fáciles de usar de la librería de tareas como bloques de construcción para ayudarle a desarrollar fácilmente el aprendizaje automático con TFLite en dispositivos móviles.

- **Procesamiento de datos complejo pero habitual** <br> Admite una lógica común de procesamiento de visión y lenguaje natural para convertir entre sus datos y el formato de datos requerido por el modelo. Ofrece la misma lógica de procesamiento, que se puede compartir, para el entrenamiento y la inferencia.

- **Ganancia de alto rendimiento** <br> El procesamiento de datos no tardaría más de unos pocos milisegundos, lo que garantiza la rápida experiencia de inferencia usando TensorFlow Lite.

- **Extensibilidad y personalización** <br> Puede aprovechar todos los beneficios que ofrece la infraestructura de la librería de tareas y crear fácilmente sus propias API de inferencia para Android/iOS.

## Tareas compatibles

A continuación figura la lista de los tipos de tareas compatibles. Se espera que la lista crezca a medida que sigamos habilitando más y más casos de uso.

- **APIs de visión**

    - [ImageClassifier](image_classifier.md)
    - [ObjectDetector](object_detector.md)
    - [ImageSegmenter](image_segmenter.md)
    - [ImageSearcher](image_searcher.md)
    - [ImageEmbedder](image_embedder.md)

- **APIs de lenguaje natural (NL)**

    - [NLClassifier](nl_classifier.md)
    - [BertNLClassifier](bert_nl_classifier.md)
    - [BertQuestionAnswerer](bert_question_answerer.md)
    - [TextSearcher](text_searcher.md)
    - [TextEmbedder](text_embedder.md)

- **APIs de audio**

    - [AudioClassifier](audio_classifier.md)

- **APIs personalizadas**

    - Amplíe la infraestructura de la API de tareas y cree [una API personalizada](customized_task_api.md).

## Ejecutar librería de tareas con delegados

Los [Delegados](https://www.tensorflow.org/lite/performance/delegates) permiten la aceleración por hardware de los modelos TensorFlow Lite aprovechando aceleradores en el dispositivo como la [GPU](https://www.tensorflow.org/lite/performance/gpu) y la [TPU Coral Edge](https://coral.ai/). Su utilización para operaciones de redes neuronales aporta enormes prestaciones en términos de latencia y eficiencia energética. Por ejemplo, las GPU pueden ofrecer hasta [5 veces más velocidad](https://blog.tensorflow.org/2020/08/faster-mobile-gpu-inference-with-opencl.html) en latencia en dispositivos móviles, y las TPU Coral Edge realizan la inferencia [10 veces más rápido](https://coral.ai/docs/edgetpu/benchmarks/) que las CPU de escritorio.

La librería de tareas ofrece una configuración sencilla y opciones de respaldo para que pueda configurar y usar delegados. Los siguientes aceleradores son ahora compatibles con la API de tareas:

- Android
    - [GPU](https://www.tensorflow.org/lite/performance/gpu): Java / C++
    - [NNAPI](https://www.tensorflow.org/lite/android/delegates/nnapi): Java / C++
    - [Hexagon](https://www.tensorflow.org/lite/android/delegates/hexagon): C++
- Linux / Mac
    - [TPU Coral Edge](https://coral.ai/): C++
- iOS
    - [Delegado Core ML](https://www.tensorflow.org/lite/performance/coreml_delegate): C++

El soporte de aceleración en Task Swift / Web API pronto estará disponible.

### Ejemplo de uso de la GPU en Android en Java

Paso 1. Añada la librería del plugin de delegado GPU al archivo `build.gradle` de su módulo:

```java
dependencies {
    // Import Task Library dependency for vision, text, or audio.

    // Import the GPU delegate plugin Library for GPU inference
    implementation 'org.tensorflow:tensorflow-lite-gpu-delegate-plugin'
}
```

Nota: NNAPI incluye por defecto los destinos de la librería de tareas para visión, texto y audio.

Paso 2. Configure el delegado de la GPU en las opciones de la tarea a través de [BaseOptions](https://www.tensorflow.org/lite/api_docs/java/org/tensorflow/lite/task/core/BaseOptions.Builder). Por ejemplo, puede configurar la GPU en `ObjectDetector` de la siguiente manera:

```java
// Turn on GPU delegation.
BaseOptions baseOptions = BaseOptions.builder().useGpu().build();
// Configure other options in ObjectDetector
ObjectDetectorOptions options =
    ObjectDetectorOptions.builder()
        .setBaseOptions(baseOptions)
        .setMaxResults(1)
        .build();

// Create ObjectDetector from options.
ObjectDetector objectDetector =
    ObjectDetector.createFromFileAndOptions(context, modelFile, options);

// Run inference
List<Detection> results = objectDetector.detect(image);
```

### Ejemplo de uso de la GPU en Android en C++

Paso 1. Cree dependencia del plugin delegado GPU en su destino de compilación bazel, como por ejemplo:

```
deps = [
  "//tensorflow_lite_support/acceleration/configuration:gpu_plugin", # for GPU
]
```

Nota: el destino `gpu_plugin` es independiente del destino [GPU delegate](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/delegates/gpu). `gpu_plugin` encapsula el destino delegado de la GPU, y puede ofrecer una protección de seguridad, es decir, respaldo a la ruta de la CPU TFLite en caso de errores de delegación.

Otras opciones de delegados son:

```
"//tensorflow_lite_support/acceleration/configuration:nnapi_plugin", # for NNAPI
"//tensorflow_lite_support/acceleration/configuration:hexagon_plugin", # for Hexagon
```

Paso 2. Configure el delegado de la GPU en las opciones de la tarea. Por ejemplo, puede configurar la GPU en `BertQuestionAnswerer` de la siguiente manera:

```c++
// Initialization
BertQuestionAnswererOptions options;
// Load the TFLite model.
auto base_options = options.mutable_base_options();
base_options->mutable_model_file()->set_file_name(model_file);
// Turn on GPU delegation.
auto tflite_settings = base_options->mutable_compute_settings()->mutable_tflite_settings();
tflite_settings->set_delegate(Delegate::GPU);
// (optional) Turn on automatical fallback to TFLite CPU path on delegation errors.
tflite_settings->mutable_fallback_settings()->set_allow_automatic_fallback_on_execution_error(true);

// Create QuestionAnswerer from options.
std::unique_ptr<QuestionAnswerer> answerer = BertQuestionAnswerer::CreateFromOptions(options).value();

// Run inference on GPU.
std::vector<QaAnswer> results = answerer->Answer(context_of_question, question_to_ask);
```

Explore ajustes más avanzados del acelerador [aquí](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/acceleration/configuration/configuration.proto).

### Ejemplo de uso de la TPU Coral Edge en Python

Configure la TPU Coral Edge en las opciones base de la tarea. Por ejemplo, puede configurar la TPU Coral Edge en `ImageClassifier` de la siguiente manera:

```python
# Imports
from tflite_support.task import vision
from tflite_support.task import core

# Initialize options and turn on Coral Edge TPU delegation.
base_options = core.BaseOptions(file_name=model_path, use_coral=True)
options = vision.ImageClassifierOptions(base_options=base_options)

# Create ImageClassifier from options.
classifier = vision.ImageClassifier.create_from_options(options)

# Run inference on Coral Edge TPU.
image = vision.TensorImage.create_from_file(image_path)
classification_result = classifier.classify(image)
```

### Ejemplo de uso de la TPU Coral Edge en C++

Paso 1. Cree dependencia del plugin delegado TPU Coral Edge en su destino de compilación bazel, como por ejemplo:

```
deps = [
  "//tensorflow_lite_support/acceleration/configuration:edgetpu_coral_plugin", # for Coral Edge TPU
]
```

Paso 2. Configure la TPU Coral Edge en las opciones de la tarea. Por ejemplo, puede configurar la TPU Coral Edge en `ImageClassifier` de la siguiente manera:

```c++
// Initialization
ImageClassifierOptions options;
// Load the TFLite model.
options.mutable_base_options()->mutable_model_file()->set_file_name(model_file);
// Turn on Coral Edge TPU delegation.
options.mutable_base_options()->mutable_compute_settings()->mutable_tflite_settings()->set_delegate(Delegate::EDGETPU_CORAL);
// Create ImageClassifier from options.
std::unique_ptr<ImageClassifier> image_classifier = ImageClassifier::CreateFromOptions(options).value();

// Run inference on Coral Edge TPU.
const ClassificationResult result = image_classifier->Classify(*frame_buffer).value();
```

Paso 3. Instale el paquete `libusb-1.0-0-dev` como se indica a continuación. Si ya está instalado, pase al siguiente paso.

```bash
# On the Linux
sudo apt-get install libusb-1.0-0-dev

# On the macOS
port install libusb
# or
brew install libusb
```

Paso 4. Compile con las siguientes configuraciones en su comando bazel:

```bash
# On the Linux
--define darwinn_portable=1 --linkopt=-lusb-1.0

# On the macOS, add '--linkopt=-lusb-1.0 --linkopt=-L/opt/local/lib/' if you are
# using MacPorts or '--linkopt=-lusb-1.0 --linkopt=-L/opt/homebrew/lib' if you
# are using Homebrew.
--define darwinn_portable=1 --linkopt=-L/opt/local/lib/ --linkopt=-lusb-1.0

# Windows is not supported yet.
```

Pruebe la [herramienta de demostración CLI de la librería de tareas](https://github.com/tensorflow/tflite-support/tree/master/tensorflow_lite_support/examples/task/vision/desktop) con sus dispositivos TPU Coral Edge. Explore más sobre los [modelos preentrenados de TPU Edge](https://coral.ai/models/) y [ajustes avanzados de TPU Edge](https://github.com/tensorflow/tensorflow/blob/4d999fda8d68adfdfacd4d0098124f1b2ea57927/tensorflow/lite/acceleration/configuration/configuration.proto#L594).

### Ejemplo de uso del delegado Core ML en C++

Puede encontrar un ejemplo completo en [Prueba de delegado Core ML del clasificador de imágenes](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/ios/test/task/vision/image_classifier/TFLImageClassifierCoreMLDelegateTest.mm).

Paso 1. Cree dependencia del plugin delegado Core ML en su destino de compilación bazel, como por ejemplo:

```
deps = [
  "//tensorflow_lite_support/acceleration/configuration:coreml_plugin", # for Core ML Delegate
]
```

Paso 2. Configure el delegado de Core ML en las opciones de la tarea. Por ejemplo, puede configurar el delegado Core ML en `ImageClassifier` de la siguiente manera:

```c++
// Initialization
ImageClassifierOptions options;
// Load the TFLite model.
options.mutable_base_options()->mutable_model_file()->set_file_name(model_file);
// Turn on Core ML delegation.
options.mutable_base_options()->mutable_compute_settings()->mutable_tflite_settings()->set_delegate(::tflite::proto::Delegate::CORE_ML);
// Set DEVICES_ALL to enable Core ML delegation on any device (in contrast to
// DEVICES_WITH_NEURAL_ENGINE which creates Core ML delegate only on devices
// with Apple Neural Engine).
options.mutable_base_options()->mutable_compute_settings()->mutable_tflite_settings()->mutable_coreml_settings()->set_enabled_devices(::tflite::proto::CoreMLSettings::DEVICES_ALL);
// Create ImageClassifier from options.
std::unique_ptr<ImageClassifier> image_classifier = ImageClassifier::CreateFromOptions(options).value();

// Run inference on Core ML.
const ClassificationResult result = image_classifier->Classify(*frame_buffer).value();
```
