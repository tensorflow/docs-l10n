# TensorFlow Lite Task ライブラリ

TensorFlow Lite Task ライブラリには、強力で使いやすいタスク固有の一連のライブラリが含まれているので、アプリ開発者はTFLite を使用して機械学習を利用できます。画像分類や質疑応答など、一般的な機械学習タスク用に最適化された、すぐに使用できるモデルインターフェイスを提供します。モデルインターフェイスは、最高のパフォーマンスと使いやすさを実現するために、タスクごとに設計されています。Task ライブラリはクロスプラットフォームで動作し、Java、C++、および Swift でサポートされています。

## Task ライブラリに期待すること

- **機械学習の専門家ではない人向けに分かりやすく明確に定義された API ** <br>推論はわずか5行のコードで実行できます。Task ライブラリの強力で使いやすい API をビルディングブロックとして使用して、モバイルデバイスで TFLite を使用して機械学習を簡単に開発できるようにします。

- **複雑かつ一般的なデータ処理** <br> モデルに必要とされるデータ形式にデータを変換するための一般的なビジョンと自然言語処理ロジックをサポートします。 トレーニングと推論のために同じ共有可能な処理ロジックを提供します。

- **高性能ゲイン** <br> データ処理にかかる時間は数ミリ秒以内なので、TensorFlow Lite を使用した高速推論が可能になります。

- **拡張性とカスタマイズ** <br> Task ライブラリインフラストラクチャが提供するすべての利点を活用して、独自の Android/iOS 推論 API を簡単に構築できます。

## サポートされているタスク

以下は、サポートされているタスクタイプのリストです。今後、ますます多くのユースケースが利用可能になり、リストに追加される予定です。

- **Vision API**

    - [ImageClassifier](image_classifier.md)
    - [ObjectDetector](object_detector.md)
    - [ImageSegmenter](image_segmenter.md)
    - [ImageSearcher](image_searcher.md)
    - [ImageEmbedder](image_embedder.md)

- **Natural Language (NL) API**

    - [NLClassifier](nl_classifier.md)
    - [BertNLClassifier](bert_nl_classifier.md)
    - [BertQuestionAnswerer](bert_question_answerer.md)
    - [TextSearcher](text_searcher.md)
    - [TextEmbedder](text_embedder.md)

- **Audio API**

    - [AudioClassifier](audio_classifier.md)

- **カスタム API**

    - Task API インフラストラクチャを拡張し、[カスタマイズされた API](customized_task_api.md) を構築します。

## デリゲートでタスクライブラリを実行する

[デリゲート](https://www.tensorflow.org/lite/performance/delegates)を使うと、[GPU](https://www.tensorflow.org/lite/performance/gpu) や [Coral Edge TPU](https://coral.ai/) などのオンデバイスアクセラレーターを利用することで、TensorFlow Lite モデルのハードウェアアクセラレーションを有効にできます。ニューラルネットワークの演算にこれらを使用すると、レイテンシーと電源効率性の観点で大きなメリットを得ることができます。たとえば、GPU の場合はモバイルデバイスで最大 [5 倍の加速化](https://blog.tensorflow.org/2020/08/faster-mobile-gpu-inference-with-opencl.html)を得られ、Coral Edge TPU の推論の場合は、デスクトップ CPU の [10 倍の速さ](https://coral.ai/docs/edgetpu/benchmarks/)を得ることができます。

Task Library provides easy configuration and fall back options for you to set up and use delegates. The following accelerators are now supported in the Task API:

- Android
    - [GPU](https://www.tensorflow.org/lite/performance/gpu): Java / C++
    - [NNAPI](https://www.tensorflow.org/lite/android/delegates/nnapi): Java / C++
    - [Hexagon](https://www.tensorflow.org/lite/android/delegates/hexagon): C++
- Linux / Mac
    - [Coral Edge TPU](https://coral.ai/): C++
- iOS
    - [Core ML delegate](https://www.tensorflow.org/lite/performance/coreml_delegate): C++

Acceleration support in Task Swift / Web API are coming soon.

### Example usage of GPU on Android in Java

Step 1. Add the GPU delegate plugin library to your module's `build.gradle` file:

```java
dependencies {
    // Import Task Library dependency for vision, text, or audio.

    // Import the GPU delegate plugin Library for GPU inference
    implementation 'org.tensorflow:tensorflow-lite-gpu-delegate-plugin'
}
```

Note: NNAPI comes with the Task Library targets for vision, text, and audio by default.

Step 2. Configure GPU delegate in the task options through [BaseOptions](https://www.tensorflow.org/lite/api_docs/java/org/tensorflow/lite/task/core/BaseOptions.Builder). For example, you can set up GPU in `ObjectDetecor` as follows:

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

ステップ 1. 次のように、bazel ビルドターゲットの GPU デリゲートプラグインに依存します。

```
deps = [
  "//tensorflow_lite_support/acceleration/configuration:gpu_plugin", # for GPU
]
```

Note: the `gpu_plugin` target is a separate one from the [GPU delegate target](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/delegates/gpu). `gpu_plugin` wraps the GPU delegate target, and can provide safety guard, i.e. fallback to TFLite CPU path on delegation errors.

その他のデリゲートオプションには、次のようなものが含まれます。

```
"//tensorflow_lite_support/acceleration/configuration:nnapi_plugin", # for NNAPI
"//tensorflow_lite_support/acceleration/configuration:hexagon_plugin", # for Hexagon
```

ステップ 2. タスクオプションに GPU デリゲートを構成します。たとえば、次のように `BertQuestionAnswerer` で GPU をセットアップできます。

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

より高度なアクセラレーター設定については、[こちら](https://github.com/tensorflow/tensorflow/blob/1a8e885b864c818198a5b2c0cbbeca5a1e833bc8/tensorflow/lite/experimental/acceleration/configuration/configuration.proto)をご覧ください。

### Example usage of Coral Edge TPU in Python

Configure Coral Edge TPU in the base options of the task. For example, you can set up Coral Edge TPU in `ImageClassifier` as follows:

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

ステップ 1. 次のように、bazel ビルドターゲットの Goral Edge TPU デリゲートプラグインに依存します。

```
deps = [
  "//tensorflow_lite_support/acceleration/configuration:edgetpu_coral_plugin", # for Coral Edge TPU
]
```

ステップ 2. タスクオプションに Coral Edge TPU を構成します。たとえば、次のように `ImageClassifier` で Coral Edge TPU をセットアップできます。

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

ステップ 3. 以下のようにして `libusb-1.0-0-dev` パッケージをインストールします。すでにインストールされている場合は、次のステップにスキップしてください。

```bash
# On the Linux
sudo apt-get install libusb-1.0-0-dev

# On the macOS
port install libusb
# or
brew install libusb
```

ステップ 4. 次の構成で bazel コマンドにコンパイルします。

```bash
# On the Linux
--define darwinn_portable=1 --linkopt=-lusb-1.0

# On the macOS, add '--linkopt=-lusb-1.0 --linkopt=-L/opt/local/lib/' if you are
# using MacPorts or '--linkopt=-lusb-1.0 --linkopt=-L/opt/homebrew/lib' if you
# are using Homebrew.
--define darwinn_portable=1 --linkopt=-L/opt/local/lib/ --linkopt=-lusb-1.0

# Windows is not supported yet.
```

Coral Edge TPU デバイスで [Task Library CLI デモツール](https://github.com/tensorflow/tflite-support/tree/master/tensorflow_lite_support/examples/task/vision/desktop)を試してみてください。[トレーニング済みの Edge TPU モデル](https://coral.ai/models/)と[高度な Edge TPU 設定](https://github.com/tensorflow/tensorflow/blob/1a8e885b864c818198a5b2c0cbbeca5a1e833bc8/tensorflow/lite/experimental/acceleration/configuration/configuration.proto#L275)をご覧ください。

### Example usage of Core ML Delegate in C++

A complete example can be found at [Image Classifier Core ML Delegate Test](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/ios/test/task/vision/image_classifier/TFLImageClassifierCoreMLDelegateTest.mm).

Step 1. Depend on the Core ML delegate plugin in your bazel build target, such as:

```
deps = [
  "//tensorflow_lite_support/acceleration/configuration:coreml_plugin", # for Core ML Delegate
]
```

Step 2. Configure Core ML Delegate in the task options. For example, you can set up Core ML Delegate in `ImageClassifier` as follows:

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
