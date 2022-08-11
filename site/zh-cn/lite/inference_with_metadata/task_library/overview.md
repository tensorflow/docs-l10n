# TensorFlow Lite Task Library

TensorFlow Lite Task Library 包含了一套功能强大且易于使用的任务专用库，供应用开发者使用 TFLite 创建机器学习体验。它为热门的机器学习任务（如图像分类、问答等）提供了经过优化的开箱即用的模型接口。模型接口专为每个任务而设计，以实现最佳性能和可用性。Task Library 可跨平台工作，支持 Java、C++ 和 Swift。

## Task Library 可以提供的内容

- **非机器学习专家也能使用的干净且定义明确的 API** <br>只需 5 行代码就可以完成推断。使用 Task Library 中强大且易用的 API 作为构建模块，帮助您在移动设备上使用 TFLite 轻松进行机器学习开发。

- **复杂但通用的数据处理** <br>支持通用的视觉和自然语言处理逻辑，可在您的数据和模型所需的数据格式之间进行转换。为训练和推断提供相同的、可共享的处理逻辑。

- **高性能增益** <br>数据处理时间不会超过几毫秒，保证了使用 TensorFlow Lite 的快速推断体验。

- **可扩展性和自定义** <br>您可以利用 Task Library 基础架构提供的所有优势，轻松构建您自己的 Android/iOS 推断 API。

## 支持的任务

以下是支持的任务类型的列表。随着我们继续提供越来越多的用例，该列表预计还会增加。

- **视觉 API**

    - [ImageClassifier](image_classifier.md)
    - [ObjectDetector](object_detector.md)
    - [ImageSegmenter](image_segmenter.md)
    - [ImageSearcher](image_searcher.md)
    - [ImageEmbedder](image_embedder.md)

- **自然语言 (NL) API**

    - [NLClassifier](nl_classifier.md)
    - [BertNLClassifier](bert_nl_classifier.md)
    - [BertQuestionAnswerer](bert_question_answerer.md)
    - [TextSearcher](text_searcher.md)
    - [TextEmbedder](text_embedder.md)

- **音频 API**

    - [AudioClassifier](audio_classifier.md)

- **自定义 API**

    - 扩展任务 API 基础架构并构建[自定义 API](customized_task_api.md)。

## 使用委托运行 Task Library

[委托](https://www.tensorflow.org/lite/performance/delegates)能够通过利用设备端加速器（如 [GPU](https://www.tensorflow.org/lite/performance/gpu) 和 [Coral Edge TPU](https://coral.ai/)）实现 TensorFlow<br>Lite 模型的硬件加速。将它们用于神经网络运算能够在延迟和功效方面提供巨大的好处。例如，GPU 可以在移动设备上提供高达 [5 倍的延迟加速](https://blog.tensorflow.org/2020/08/faster-mobile-gpu-inference-with-opencl.html)，而 Coral Edge TPU 推断比桌面电脑 CPU [快 10 倍](https://coral.ai/docs/edgetpu/benchmarks/)。

Task Library 为您设置和使用委托提供了简单的配置和后备选项。Task API 目前支持以下加速器：

- Android
    - [GPU](https://www.tensorflow.org/lite/performance/gpu): Java / C++
    - [NNAPI](https://www.tensorflow.org/lite/android/delegates/nnapi): Java / C++
    - [Hexagon](https://www.tensorflow.org/lite/android/delegates/hexagon): C++
- Linux / Mac
    - [Coral Edge TPU](https://coral.ai/): C++
- iOS
    - [Core ML delegate](https://www.tensorflow.org/lite/performance/coreml_delegate): C++

Task Swift / Web API 中的加速支持即将推出。

### 用 Java 实现 Android 平台上 GPU 使用的示例

第 1 步：将 GPU 委托插件库添加到您模块的 `build.gradle` 文件：

```java
dependencies {
    // Import Task Library dependency for vision, text, or audio.

    // Import the GPU delegate plugin Library for GPU inference
    implementation 'org.tensorflow:tensorflow-lite-gpu-delegate-plugin'
}
```

注：默认情况下，NNAPI 附带针对视觉、文本和音频的 Task Library。

第 2 步：通过 [BaseOptions](https://www.tensorflow.org/lite/api_docs/java/org/tensorflow/lite/task/core/BaseOptions.Builder) 在任务选项中配置 GPU 委托。例如，您可以在 `ObjectDetecor` 中设置 GPU，如下所示：

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

### 用 C++ 实现 Android 平台上 GPU 使用的示例

第 1 步：依赖于 Bazel 构建目标中的 GPU 委托插件，例如：

```
deps = [
  "//tensorflow_lite_support/acceleration/configuration:gpu_plugin", # for GPU
]
```

注：`gpu_plugin` 目标与 [GPU 委托目标](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/delegates/gpu) 不同。 `gpu_plugin` 封装了 GPU 委托目标，可以提供安全防护，即委托出错时回退到 TFLite CPU 路径。

其他委托选项包括：

```
"//tensorflow_lite_support/acceleration/configuration:nnapi_plugin", # for NNAPI
"//tensorflow_lite_support/acceleration/configuration:hexagon_plugin", # for Hexagon
```

第 2 步：在任务选项中配置 GPU 委托。例如，您可以在 `BertQuestionAnswerer` 中设置 GPU，如下所示：

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

在[此处](https://github.com/tensorflow/tensorflow/blob/1a8e885b864c818198a5b2c0cbbeca5a1e833bc8/tensorflow/lite/experimental/acceleration/configuration/configuration.proto)浏览更多高级加速器设置。

### 用 Python 实现 Coral Edge TPU 使用的示例

在任务的基本选项中配置 Coral Edge TPU。例如，您可以在 `ImageClassifier` 中设置 Coral Edge TPU，如下所示：

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

### 用 C++ 实现 Coral Edge TPU 使用的示例

第 1 步：依赖 Bazel 构建目标中的 Coral Edge TPU 委托插件，例如：

```
deps = [
  "//tensorflow_lite_support/acceleration/configuration:edgetpu_coral_plugin", # for Coral Edge TPU
]
```

第 2 步：在任务选项中配置 Coral Edge TPU。例如，您可以在 `ImageClassifier` 中设置 Coral Edge TPU，如下所示：

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

第 3 步：使用如下代码安装 `libusb-1.0-0-dev` 软件包。如果已经安装，请跳到下一步。

```bash
# On the Linux
sudo apt-get install libusb-1.0-0-dev

# On the macOS
port install libusb
# or
brew install libusb
```

第 4 步：在 Bazel 命令中使用以下配置进行编译：

```bash
# On the Linux
--define darwinn_portable=1 --linkopt=-lusb-1.0

# On the macOS, add '--linkopt=-lusb-1.0 --linkopt=-L/opt/local/lib/' if you are
# using MacPorts or '--linkopt=-lusb-1.0 --linkopt=-L/opt/homebrew/lib' if you
# are using Homebrew.
--define darwinn_portable=1 --linkopt=-L/opt/local/lib/ --linkopt=-lusb-1.0

# Windows is not supported yet.
```

在您的 Coral Edge TPU 设备上试用 [Task Library CLI 演示工具](https://github.com/tensorflow/tflite-support/tree/master/tensorflow_lite_support/examples/task/vision/desktop)。了解更多有关[预训练的 Edge TPU 模型](https://coral.ai/models/)和[高级 Edge TPU 设置](https://github.com/tensorflow/tensorflow/blob/1a8e885b864c818198a5b2c0cbbeca5a1e833bc8/tensorflow/lite/experimental/acceleration/configuration/configuration.proto#L275)的详细信息。

### 用 C++ 实现 Core ML Delegate 使用的示例

完整示例可在[图像分类器核心机器学习委托测试](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/ios/test/task/vision/image_classifier/TFLImageClassifierCoreMLDelegateTest.mm)中找到。

第 1 步：依赖 Bazel 构建目标中的 Core ML 委托插件，例如：

```
deps = [
  "//tensorflow_lite_support/acceleration/configuration:coreml_plugin", # for Core ML Delegate
]
```

第 2 步：在任务选项中配置 Core ML Delegate。例如，您可以在 `ImageClassifier` 中设置 Core ML Delegate，如下所示：

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
