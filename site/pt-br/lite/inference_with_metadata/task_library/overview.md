# Biblioteca Task do TensorFlow Lite

A biblioteca Task do TensorFlow Lite contém um conjunto de bibliotecas específicas a tarefas poderoso e fácil de usar para os desenvolvedores de apps criarem experiências de aprendizado de máquina com o TF Lite. Ela fornece interfaces de modelo prontas para uso e otimizadas para tarefas de aprendizado de máquina populares, como classificação de imagens, pergunta e resposta, etc. As interfaces de modelo são criadas especialmente para cada tarefa alcançar o melhor desempenho e usabilidade. A biblioteca de tarefas funciona em várias plataformas e é compatível com o Java, C++ e Swift.

## O que esperar da Task Library

- **APIs organizadas e bem definidas, que podem ser utilizadas por quem não é especialista em aprendizado de máquina** <br> A inferência pode ser feita com apenas 5 linhas de código. Utilize as APIs avançadas e fáceis de usar da biblioteca Task como blocos de construção para desenvolver aprendizado de máquina com o TF Lite facilmente em dispositivos móveis.

- **Processamento de dados complexo, mas comum** <br> A biblioteca oferece suporte a lógicas comuns de processamento de linguagem natural e visão para converter entre seus dados e o formato de dados exigido pelo modelo. Oferece a mesma lógica de processamento compartilhável para treinamento e inferência.

- **Alto ganho de desempenho** <br> O processamento dos dados pode levar apena alguns milissegundos, garantindo uma inferência rápida ao usar o TensorFlow Lite.

- **Extensibilidade e personalização** <br> Você pode aproveitar todos os benefícios oferecidos pela infraestrutura da biblioteca de tarefas e compilar facilmente suas próprias APIs de inferência para Android e iOS.

## Tarefas com suporte

Veja abaixo a lista de tarefas com suporte. Essa lista deverá crescer à medida que adicionarmos mais casos de uso.

- **APIs de visão**

    - [ImageClassifier](image_classifier.md)
    - [ObjectDetector](object_detector.md)
    - [ImageSegmenter](image_segmenter.md)
    - [ImageSearcher](image_searcher.md)
    - [ImageEmbedder](image_embedder.md)

- **APIs de linguagem natural (NL)**

    - [NLClassifier](nl_classifier.md)
    - [BertNLClassifier](bert_nl_classifier.md)
    - [BertQuestionAnswerer](bert_question_answerer.md)
    - [TextSearcher](text_searcher.md)
    - [TextEmbedder](text_embedder.md)

- **APIs de áudio**

    - [AudioClassifier](audio_classifier.md)

- **APIs personalizadas**

    - Estenda a infraestrutura da API de tarefas e crie [APIs personalizadas](customized_task_api.md).

## Execute a biblioteca Task com delegados

Os [delegados](https://www.tensorflow.org/lite/performance/delegates) possibilitam a aceleração de hardware para modelos do TensorFlow Lite por meio do uso de aceleradores no dispositivo, como [GPU](https://www.tensorflow.org/lite/performance/gpu) e [Coral Edge TPU](https://coral.ai/). Ao usá-los para operações de redes neurais, há diversos benefícios com relação à latência e à eficiência energética. Por exemplo: as GPUs podem oferecer uma [melhoria de até 5 vezes](https://blog.tensorflow.org/2020/08/faster-mobile-gpu-inference-with-opencl.html) da latência em dispositivos móveis e inferência em Coral Edge TPUs [10 vezes mais rápida](https://coral.ai/docs/edgetpu/benchmarks/) do que em CPUs de computadores.

A biblioteca Task conta com fácil configuração e opções de fallback para o uso de delegados. Atualmente, a API de tarefas oferece suporte aos seguintes aceleradores:

- Android
    - [GPU](https://www.tensorflow.org/lite/performance/gpu): Java/C++
    - [NNAPI](https://www.tensorflow.org/lite/android/delegates/nnapi): Java/C++
    - [Hexagon](https://www.tensorflow.org/lite/android/delegates/hexagon): C++
- Linux/Mac
    - [Coral Edge TPU](https://coral.ai/): C++
- iOS
    - [Delegado Core ML](https://www.tensorflow.org/lite/performance/coreml_delegate): C++

Em breve, teremos suporte à aceleração na API de tarefas para Swift/Web.

### Exemplo de uso de GPU no Android no Java

Etapa 1 – Adicione a biblioteca de plug-in de delegado GPU ao arquivo `build.gradle` do seu módulo:

```java
dependencies {
    // Import Task Library dependency for vision, text, or audio.

    // Import the GPU delegate plugin Library for GPU inference
    implementation 'org.tensorflow:tensorflow-lite-gpu-delegate-plugin'
}
```

Observação: por padrão, a NNAPI vem com os alvos da biblioteca Task para visão, texto e áudio.

Etapa 2 – Configure o delegado nas opções da tarefa com as [BaseOptions](https://www.tensorflow.org/lite/api_docs/java/org/tensorflow/lite/task/core/BaseOptions.Builder). Por exemplo: você pode configurar a GPU em `ObjectDetector` da seguinte forma:

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

### Exemplo de uso de GPU no Android no C++

Etapa 1 – Adicione a dependência do plug-in de delegado GPU ao alvo da build do Bazel da seguinte forma:

```
deps = [
  "//tensorflow_lite_support/acceleration/configuration:gpu_plugin", # for GPU
]
```

Observação: o alvo `gpu_plugin` fica separado do [alvo de delegado GPU](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/delegates/gpu). `gpu_plugin` encapsula o alvo de delegado GPU e pode proporcionar uma proteção, ou seja, fazer o fallback para o caminho de CPU do TF Lite em caso de erros de delegação.

Confira outras opções de delegado:

```
"//tensorflow_lite_support/acceleration/configuration:nnapi_plugin", # for NNAPI
"//tensorflow_lite_support/acceleration/configuration:hexagon_plugin", # for Hexagon
```

Etapa 2 – Configure o delegado nas opções da tarefa. Por exemplo: você pode configurar a GPU em `BertQuestionAnswerer` da seguinte forma:

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

Veja configurações de acelerador mais avançadas [aqui](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/acceleration/configuration/configuration.proto).

### Exemplo de uso do Coral Edge TPU no Python

Configure o Coral Edge TPU nas opções base da tarefa. Por exemplo: você pode configurar o Coral Edge TPU em `ImageClassifier` da seguinte forma:

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

### Exemplo de uso do Coral Edge TPU no C++

Etapa 1 – Adicione a dependência do plug-in de delegado Coral Edge TPU ao alvo da build do Bazel da seguinte forma:

```
deps = [
  "//tensorflow_lite_support/acceleration/configuration:edgetpu_coral_plugin", # for Coral Edge TPU
]
```

Configure o Coral Edge TPU nas opções da tarefa. Por exemplo: você pode configurar o Coral Edge TPU em `ImageClassifier` da seguinte forma:

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

Etapa 3 – Instale o pacote `libusb-1.0-0-dev` conforme indicado abaixo. Caso já esteja instalado, prossiga para a próxima etapa.

```bash
# On the Linux
sudo apt-get install libusb-1.0-0-dev

# On the macOS
port install libusb
# or
brew install libusb
```

Etapa 4 – Compile com as configurações abaixo no comando do Bazel:

```bash
# On the Linux
--define darwinn_portable=1 --linkopt=-lusb-1.0

# On the macOS, add '--linkopt=-lusb-1.0 --linkopt=-L/opt/local/lib/' if you are
# using MacPorts or '--linkopt=-lusb-1.0 --linkopt=-L/opt/homebrew/lib' if you
# are using Homebrew.
--define darwinn_portable=1 --linkopt=-L/opt/local/lib/ --linkopt=-lusb-1.0

# Windows is not supported yet.
```

Experimente a [ferramenta CLI de demonstração da biblioteca Task](https://github.com/tensorflow/tflite-support/tree/master/tensorflow_lite_support/examples/task/vision/desktop) em seus dispositivos Coral Edge TPU. Veja mais detalhes nos [modelos pré-treinados para Edge TPU](https://coral.ai/models/) e nas [configurações avançadas de Edge TPU](https://github.com/tensorflow/tensorflow/blob/4d999fda8d68adfdfacd4d0098124f1b2ea57927/tensorflow/lite/acceleration/configuration/configuration.proto#L594).

### Exemplo de uso de delegado Core ML no C++

Veja um exemplo completo no [teste de delegado Core ML para classificador de imagem](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/ios/test/task/vision/image_classifier/TFLImageClassifierCoreMLDelegateTest.mm).

Etapa 1 – Adicione a dependência do plug-in de delegado Core ML ao alvo da build do Bazel da seguinte forma:

```
deps = [
  "//tensorflow_lite_support/acceleration/configuration:coreml_plugin", # for Core ML Delegate
]
```

Etapa 2 – Configure o delegado Core ML nas opções da tarefa. Por exemplo: você pode configurar o delegado Core ML em `ImageClassifier` da seguinte forma:

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
