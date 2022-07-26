# TensorFlow Lite Task Library

TensorFlow Lite Task Library contains a set of powerful and easy-to-use
task-specific libraries for app developers to create ML experiences with TFLite.
It provides optimized out-of-box model interfaces for popular machine learning
tasks, such as image classification, question and answer, etc. The model
interfaces are specifically designed for each task to achieve the best
performance and usability. Task Library works cross-platform and is supported on
Java, C++, and Swift.

## What to expect from the Task Library

*   **Clean and well-defined APIs usable by non-ML-experts** \
    Inference can be done within just 5 lines of code. Use the powerful and
    easy-to-use APIs in the Task library as building blocks to help you easily
    develop ML with TFLite on mobile devices.

*   **Complex but common data processing** \
    Supports common vision and natural language processing logic to convert
    between your data and the data format required by the model. Provides the
    same, shareable processing logic for training and inference.

*   **High performance gain** \
    Data processing would take no more than a few milliseconds, ensuring the
    fast inference experience using TensorFlow Lite.

*   **Extensibility and customization** \
    You can leverage all benefits the Task Library infrastructure provides and
    easily build your own Android/iOS inference APIs.

## Supported tasks

Below is the list of the supported task types. The list is expected to grow as
we continue enabling more and more use cases.

*   **Vision APIs**

    *   [ImageClassifier](image_classifier.md)
    *   [ObjectDetector](object_detector.md)
    *   [ImageSegmenter](image_segmenter.md)
    *   [ImageSearcher](image_searcher.md)
    *   [ImageEmbedder](image_embedder.md)

*   **Natural Language (NL) APIs**

    *   [NLClassifier](nl_classifier.md)
    *   [BertNLClassifier](bert_nl_classifier.md)
    *   [BertQuestionAnswerer](bert_question_answerer.md)
    *   [TextSearcher](text_searcher.md)
    *   [TextEmbedder](text_embedder.md)

*   **Audio APIs**

    *   [AudioClassifier](audio_classifier.md)

*   **Custom APIs**

    *   Extend Task API infrastructure and build
        [customized API](customized_task_api.md).

## Run Task Library with Delegates

[Delegates](https://www.tensorflow.org/lite/performance/delegates) enable
hardware acceleration of TensorFlow Lite models by leveraging on-device
accelerators such as the [GPU](https://www.tensorflow.org/lite/performance/gpu)
and [Coral Edge TPU](https://coral.ai/). Utilizing them for neural network
operations provides huge benefits in terms of latency and power efficiency. For
example, GPUs can provide upto a
[5x speedup](https://blog.tensorflow.org/2020/08/faster-mobile-gpu-inference-with-opencl.html)
in latency on mobile devices, and Coral Edge TPUs inference
[10x faster](https://coral.ai/docs/edgetpu/benchmarks/) than desktop CPUs.

Task Library provides easy configuration and fall back options for you to set up
and use delegates. The following accelerators are now supported in the Task API:

*   Android
    *   [GPU](https://www.tensorflow.org/lite/performance/gpu): Java / C++
    *   [NNAPI](https://www.tensorflow.org/lite/android/delegates/nnapi):
        Java / C++
    *   [Hexagon](https://www.tensorflow.org/lite/android/delegates/hexagon):
        C++
*   Linux / Mac
    *   [Coral Edge TPU](https://coral.ai/): C++
*   iOS
    *   [Core ML delegate](https://www.tensorflow.org/lite/performance/coreml_delegate):
        C++

Acceleration support in Task Swift / Web API are coming soon.

### Example usage of GPU on Android in Java

Step 1. Add the GPU delegate plugin library to your module's `build.gradle`
file:

```java
dependencies {
    // Import Task Library dependency for vision, text, or audio.

    // Import the GPU delegate plugin Library for GPU inference
    implementation 'org.tensorflow:tensorflow-lite-gpu-delegate-plugin'
}
```

Note: NNAPI comes with the Task Library targets for vision, text, and audio by
default.

Step 2. Configure GPU delegate in the task options through
[BaseOptions](https://www.tensorflow.org/lite/api_docs/java/org/tensorflow/lite/task/core/BaseOptions.Builder).
For example, you can set up GPU in `ObjectDetecor` as follows:

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

### Example usage of GPU on Android in C++

Step 1. Depend on the GPU delegate plugin in your bazel build target, such as:

```
deps = [
  "//tensorflow_lite_support/acceleration/configuration:gpu_plugin", # for GPU
]
```

Note: the `gpu_plugin` target is a separate one from the
[GPU delegate target](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/delegates/gpu).
`gpu_plugin` wraps the GPU delegate target, and can provide safety guard, i.e.
fallback to TFLite CPU path on delegation errors.

Other delegate options include:

```
"//tensorflow_lite_support/acceleration/configuration:nnapi_plugin", # for NNAPI
"//tensorflow_lite_support/acceleration/configuration:hexagon_plugin", # for Hexagon
```

Step 2. Configure GPU delegate in the task options. For example, you can set up
GPU in `BertQuestionAnswerer` as follows:

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

Explore more advanced accelerator settings
[here](https://github.com/tensorflow/tensorflow/blob/1a8e885b864c818198a5b2c0cbbeca5a1e833bc8/tensorflow/lite/experimental/acceleration/configuration/configuration.proto).

### Example usage of Coral Edge TPU in Python

Configure Coral Edge TPU in the base options of the task. For example, you can
set up Coral Edge TPU in `ImageClassifier` as follows:

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

### Example usage of Coral Edge TPU in C++

Step 1. Depend on the Coral Edge TPU delegate plugin in your bazel build target,
such as:

```
deps = [
  "//tensorflow_lite_support/acceleration/configuration:edgetpu_coral_plugin", # for Coral Edge TPU
]
```

Step 2. Configure Coral Edge TPU in the task options. For example, you can set
up Coral Edge TPU in `ImageClassifier` as follows:

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

Step 3. Install the `libusb-1.0-0-dev` package as below. If it is already
installed, skip to the next step.

```bash
# On the Linux
sudo apt-get install libusb-1.0-0-dev

# On the macOS
port install libusb
# or
brew install libusb
```

Step 4. Compile with the following configurations in your bazel command:

```bash
# On the Linux
--define darwinn_portable=1 --linkopt=-lusb-1.0

# On the macOS, add '--linkopt=-lusb-1.0 --linkopt=-L/opt/local/lib/' if you are
# using MacPorts or '--linkopt=-lusb-1.0 --linkopt=-L/opt/homebrew/lib' if you
# are using Homebrew.
--define darwinn_portable=1 --linkopt=-L/opt/local/lib/ --linkopt=-lusb-1.0

# Windows is not supported yet.
```

Try out the
[Task Library CLI demo tool](https://github.com/tensorflow/tflite-support/tree/master/tensorflow_lite_support/examples/task/vision/desktop)
with your Coral Edge TPU devices. Explore more on the
[pretrained Edge TPU models](https://coral.ai/models/) and
[advanced Edge TPU settings](https://github.com/tensorflow/tensorflow/blob/1a8e885b864c818198a5b2c0cbbeca5a1e833bc8/tensorflow/lite/experimental/acceleration/configuration/configuration.proto#L275).

### Example usage of Core ML Delegate in C++

A complete example can be found at
[Image Classifier Core ML Delegate Test](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/ios/test/task/vision/image_classifier/TFLImageClassifierCoreMLDelegateTest.mm).

Step 1. Depend on the Core ML delegate plugin in your bazel build target, such
as:

```
deps = [
  "//tensorflow_lite_support/acceleration/configuration:coreml_plugin", # for Core ML Delegate
]
```

Step 2. Configure Core ML Delegate in the task options. For example, you can set
up Core ML Delegate in `ImageClassifier` as follows:

```c++
// Initialization
ImageClassifierOptions options;
// Load the TFLite model.
options.mutable_base_options()->mutable_model_file()->set_file_name(model_file);
// Turn on Core ML delegation.
options.mutable_base_options()->mutable_compute_settings()->mutable_tflite_settings()->set_delegate(Delegate::CORE_ML);
// Set DEVICES_ALL to enable Core ML delegation on any device (in contrast to
// DEVICES_WITH_NEURAL_ENGINE which creates Core ML delegate only on devices
// with Apple Neural Engine).
options.mutable_base_options()->mutable_compute_settings()->mutable_tflite_settings()->mutable_coreml_settings()->set_enabled_devices(CoreMLDelegateSettings::DEVICES_ALL);
// Create ImageClassifier from options.
std::unique_ptr<ImageClassifier> image_classifier = ImageClassifier::CreateFromOptions(options).value();

// Run inference on Core ML.
const ClassificationResult result = image_classifier->Classify(*frame_buffer).value();
```
